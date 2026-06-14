"""
Apollo 11 crew survival model.

Maps each mission failure mode to a PER-ASTRONAUT crew-survival outcome based
on the Apollo abort architecture and per-crewman exposure (Collins in the CSM
vs the two moonwalkers). Adds these columns to the results CSV, for each of
Armstrong / Aldrin / Collins:

  survived_<name>     — Boolean (random draw)
  psurv_<name>        — survival probability used for that astronaut
  dcause_<name>       — what killed that astronaut ('' if survived; the trial's
                        mission_failure, or crew_return_leg_loss for a death on
                        the Earth-return leg after surviving the precipitating event)

plus aggregate columns:

  n_astronauts_survived — 0..3 for the trial
  crew_survived         — Boolean, all three survived
  crew_survival_prob    — min individual survival probability
  crew_outcome          — categorical label for the trial

Apollo had a well-defined abort architecture covering most failure modes:

  Mode I    (pad → 30s post-S-II ignition):  LES (Launch Escape System)
                                              pulls CM clear; parachute landing
  Mode II   (S-II to S-IVB cutoff):           CM splash in N. Atlantic via LES
  Mode III  (high-altitude S-II):             Lower-velocity abort
  Mode IV   (S-IVB failure):                  SPS burns to put CSM in EPO
                                              (Earth Parking Orbit)
  COI       (post-SECO):                      Continue in parking orbit; can
                                              return any time via SPS deorbit
  Free-return  (TLI to LOI):                  Trajectory swings around Moon
                                              and returns to Earth (Apollo 13)
  Abort-Stage (descent):                      LM ascent stage ignites; leaves
                                              descent stage behind
  Stranded on Moon (ascent failure):          No return path — "the speech"

References:
  - Apollo 12 Pre-launch Mission Operation Report (NASA HQ, 1969):
    https://www.nasa.gov/wp-content/uploads/static/history/afj/ap12fj/
    pdf/a12-prelaunch-rep2.pdf
  - Apollo 13 "successful failure" — free-return trajectory saved crew
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = "outputs/apollo11_final10000"
CREW_SIZE = 3   # Armstrong, Aldrin, Collins

# Roster and crew roles. Survival diverges by VEHICLE during the lunar phase:
#   - Collins (CMP) stays in the CSM in lunar orbit; he is NOT exposed to
#     descent/surface/ascent/rendezvous risk, and if the LM crew is lost his
#     documented contingency is to return to Earth ALONE.
#   - Armstrong (CDR) and Aldrin (LMP) ride the LM to the surface and back;
#     they are exposed to every risk Collins faces PLUS the LM phases.
# HONESTY NOTE: Armstrong and Aldrin share a vehicle for every phase EXCEPT a
# terminal EVA suit/PLSS failure (surface_eva_suit_fatality), which kills
# exactly one of them (the suit-failure victim, chosen at random). That is the
# ONLY modeled individual-level risk, so their survival rates differ only by
# the handful of EVA-fatality trials (~0.1%/mission rate) — a near-noise-level
# difference, not a structural one. Every other LM-phase mode is a shared
# fate. The large, meaningful distinction remains Collins (CSM, a strict
# subset of the LM crew's risks) vs the two moonwalkers.
CREW = ["Armstrong", "Aldrin", "Collins"]
LM_CREW = ["Armstrong", "Aldrin"]   # descend in the LM
CSM_CREW = ["Collins"]              # stays in the CSM

# Which crew each failure PHASE exposes:
#   "all"     — the whole stack shares the fate (launch, translunar, TEI,
#               trans-Earth coast, entry — everyone is in the same vehicle).
#   "lm_crew" — only Armstrong & Aldrin are at risk; Collins is safe in the
#               CSM and must still make a (solo) Earth return.
def crew_exposed_for_failure(fail_mode):
    if fail_mode is None or (isinstance(fail_mode, float)):
        return "all"   # nominal — handled separately
    f = str(fail_mode)
    # A terminal EVA suit/PLSS failure kills exactly ONE moonwalker (the
    # astronaut whose suit failed); the partner and Collins return. This is
    # the ONLY modeled mode that distinguishes Armstrong from Aldrin — every
    # other LM-phase mode is a shared-vehicle fate.
    if f == "surface_eva_suit_fatality":
        return "one_lm_crew"
    if (f.startswith("descent_") or f.startswith("ascent_")
            or f.startswith("rendezvous_") or f.startswith("surface_")):
        return "lm_crew"
    # launch_*, missed_lunar_soi*, tei_*, transearth_*, entry_* → whole stack
    return "all"


# Survival model: maps failure mode → (P_survival, outcome_label,
# explanation of recovery path).
#
# Probabilities are estimates. IMPORTANT: Apollo flew the crewed LM only nine
# times with ZERO descent/ascent/rendezvous fatalities, so there is no observed
# (frequentist) failure rate for these phases — every "rate" here is an
# engineering/expert-judgment estimate, NOT a measured frequency. The estimates
# are anchored to the best available NASA sources:
#  - NASA Lunar Landing Operational Risk Model (LLORM, Mattenberger et al.,
#    NTRS 20100018589): a Monte-Carlo lunar-landing risk model. Gives expert-
#    judgment values used directly below — nominal-abort success P(S)=0.999,
#    emergency (time-critical) abort success P(S)=0.9, and successful-touchdown
#    P(S)=0.999. It also establishes that landing risk is driven by OPERATIONAL
#    factors (terrain, navigation dispersion, propellant margin, crew decision
#    time), NOT random hardware failure — which is why propellant-margin failure
#    dominates this model's LM phase rather than engine failures.
#  - 1965 NASA Apollo risk assessment: per-flight mission-success ~73%, rated
#    per-mission crew safety ~96% (order-of-magnitude anchor for crew survival).
#  - Apollo abort architecture documentation; LES reliability (~98%); SPS
#    reliability and propellant margin; Apollo 13 free-return precedent.
#  - 1201/1202 alarms (Apollo 11): a recoverable design/procedural event (radar
#    switch), five alarms all recovered — modeled as recoverable, not a random
#    hardware hazard (Discover/Draper accounts, NTRS exegesis).
SURVIVAL_MODEL = {
    "launch_structural_failure_max_q_exceeded": {
        "p_survive": 0.30,
        "outcome": "abort_mode_I_LES_late",
        "explanation": (
            "Vehicle breakup during transonic flight (~50-70s after liftoff). "
            "The LES (Launch Escape System) is designed for exactly this — but "
            "max-Q breakup is violent and the LES requires the CM to still be "
            "intact when LES fires. Historical precedent: no Apollo vehicle "
            "ever had max-Q breakup, so this is a worst-case estimate. The "
            "30% reflects: ~98% LES nominal reliability × ~30% chance the CM "
            "is still intact and oriented favorably after the breakup event."
        ),
    },
    "launch_s_ic_underperformance_crash": {
        "p_survive": 0.85,
        "outcome": "abort_mode_I_LES",
        "explanation": (
            "S-IC underperformance is a SLOW failure — the rocket loses "
            "altitude over many seconds. EDS (Emergency Detection System) or "
            "crew commands LES Mode I: solid-fuel escape motor pulls CM clear "
            "of the stack, deploys parachutes, splashes down in Atlantic. "
            "Apollo Pad Abort Test 1 & 2 validated this in 1963-65. The 85% "
            "reflects LES nominal reliability minus edge cases (LES misfire, "
            "parachute failure)."
        ),
    },
    "launch_s_ivb_first_ignition_failure": {
        "p_survive": 0.95,
        "outcome": "abort_mode_IV_COI",
        "explanation": (
            "S-IVB failed to ignite after S-II burnout. This is exactly the "
            "scenario Mode IV (Contingency Orbit Insertion) was designed for: "
            "CSM separates, SPS burns to inject into a stable Earth parking "
            "orbit, then performs a normal entry and splashdown. SPS has "
            "ample fuel — used only for lunar maneuvers in nominal mission. "
            "Crew survives easily."
        ),
    },
    "launch_parking_orbit_decays_into_atmosphere": {
        "p_survive": 0.85,
        "outcome": "abort_SPS_deorbit",
        "explanation": (
            "Perigee below 80 km — orbit will decay in hours to days, but "
            "crew has time. CSM separates from S-IVB and uses SPS retrograde "
            "burn to set up controlled entry. The 85% reflects the additional "
            "risk that the unstable orbit might force entry at a bad flight "
            "path angle if SPS can't be commanded in time."
        ),
    },
    "launch_parking_insertion_overshoot": {
        "p_survive": 0.92,
        "outcome": "abort_SPS_circularize",
        "explanation": (
            "Apogee above 1000 km but orbit is energetic enough — CSM uses "
            "SPS to circularize or deorbit. Slight risk from extended "
            "exposure to radiation belts if apogee > 2000 km."
        ),
    },
    "launch_parking_insertion_overshoot_escape": {
        "p_survive": 0.10,
        "outcome": "abort_failed_escape_trajectory",
        "explanation": (
            "S-IVB overburned into hyperbolic Earth escape. With CSM "
            "propellant designed for lunar mission, recovery to Earth orbit "
            "may be impossible if escape velocity was significantly exceeded."
        ),
    },
    "missed_lunar_soi": {
        "p_survive": 0.98,
        "outcome": "free_return_trajectory",
        "explanation": (
            "Trans-lunar trajectory missed Moon SOI. Apollo's mission design "
            "deliberately used a free-return trajectory: even without LOI, "
            "the spacecraft would swing around the Moon and return to Earth "
            "atmosphere on a survivable entry trajectory. This is precisely "
            "how Apollo 13's crew survived after their SM oxygen tank "
            "explosion. The 98% reflects this proven recovery path; the 2% "
            "covers rare cases where the periapsis would have been far below "
            "the Moon's surface (impact) rather than just outside the "
            "recoverable range."
        ),
    },
    "descent_descent_propellant_exhausted": {
        "p_survive": 0.70,
        "outcome": "abort_stage_LM_ascent_engine",
        "explanation": (
            "LM ran out of DPS propellant during powered descent. Apollo's "
            "'abort stage' procedure ignites the APS (ascent engine) while "
            "the descent stage is still attached, separating in-flight and "
            "putting the ascent stage into lunar orbit for CSM rendezvous. "
            "Effectiveness depends on altitude and descent rate at fuel "
            "exhaustion: high altitude / low descent rate → high success; "
            "close to surface at high vertical velocity → no time to abort. "
            "The 70% reflects the typical distribution of altitudes/speeds "
            "at fuel exhaustion in this simulation."
        ),
    },
    "descent_breakup_on_landing": {
        "p_survive": 0.20,
        "outcome": "hard_landing_LM_damaged",
        "explanation": (
            "Touchdown velocity exceeded LM landing gear design tolerance "
            "(~3 m/s vertical, ~1.5 m/s horizontal). The LM struck the "
            "surface hard enough to damage the structure. Survival possible "
            "only if damage was localized and ascent stage remained "
            "functional for return."
        ),
    },
    "ascent_engine_failure": {
        "p_survive": 0.0,
        "outcome": "stranded_on_lunar_surface",
        "explanation": (
            "The Lunar Module's ascent engine (APS) failed to ignite or "
            "produced insufficient ΔV to reach lunar orbit. The crew is "
            "stranded on the Moon's surface with no return path. This is "
            "the scenario for which Nixon's speechwriter William Safire "
            "drafted 'In Event of Moon Disaster' — the speech that never "
            "had to be given."
        ),
    },
    "ascent_insufficient_dv": {
        "p_survive": 0.0,
        "outcome": "stranded_on_lunar_surface",
        "explanation": (
            "Ascent burn completed but failed to reach a stable lunar orbit. "
            "LM either fell back to the surface or entered an unstable orbit "
            "with no recovery."
        ),
    },
    "rendezvous_insufficient_propellant": {
        "p_survive": 0.0,
        "outcome": "stranded_in_lunar_orbit",
        "explanation": (
            "After ascent the LM lacked the propellant to null the orbital "
            "plane/altitude mismatch with the CSM and close for docking "
            "(driven mainly by an ascent-insertion plane error). Armstrong and "
            "Aldrin are stranded in lunar orbit within sight of the CSM but "
            "unable to reach it. Collins, in the healthy CSM, returns to Earth "
            "alone. Exposure is LM-crew-only."
        ),
    },
    "rendezvous_docking_failure": {
        "p_survive": 0.90,
        "outcome": "eva_crew_transfer",
        "explanation": (
            "The LM closed with the CSM but the docking mechanism failed "
            "unrecoverably. SOURCED RATE: capture anomalies on 2 of ~21 "
            "program dockings (Apollo 14's six attempts; Skylab 2's "
            "interlock bypass), times ~10% chance the independent ring-latch "
            "hard-dock workaround also fails — ~0.95% unrecovered per "
            "docking. The documented contingency is a suited EVA crew "
            "transfer from the LM to the CSM side hatch, modeled at ~90% "
            "survival (estimate — planned and trained, never flown). "
            "Exposure is LM-crew-only; Collins is in the healthy CSM."
        ),
    },
    "transposition_docking_failure": {
        "p_survive": 0.99,
        "outcome": "landing_aborted_healthy_return",
        "explanation": (
            "Transposition & docking after TLI failed unrecoverably — the LM "
            "cannot be extracted from the S-IVB, the landing is aborted, and "
            "the crew returns on a fully healthy CSM (free-return/direct "
            "abort; this nearly ended Apollo 14's landing before its "
            "hard-dock workaround succeeded). Mission failure, but survival "
            "is essentially that of a normal return (~99%). Same sourced "
            "~0.95% per-docking rate as the rendezvous docking."
        ),
    },
    "surface_lm_tipover": {
        "p_survive": 0.0,
        "outcome": "lm_tipped_no_ascent",
        "explanation": (
            "The LM tipped beyond recovery at touchdown (0-in-6 landings "
            "historically against a ~12 deg stability spec; Apollo 15's 11 "
            "deg was the worst case). No ascent is possible from a tipped "
            "LM and no rescue capability exists. LM crew lost; Collins "
            "returns alone."
        ),
    },
    "surface_eva_suit_fatality": {
        "p_survive": 0.5,
        "outcome": "eva_fatality_partner_returns",
        "explanation": (
            "A terminal suit/PLSS failure during the surface EVA kills one "
            "of the two moonwalkers (0 failures in 28 program man-EVAs; the "
            "fatal branch requires the anomaly to outrun the OPS backup and "
            "an immediate LM repress, within ~60 m of the hatch). Modeled "
            "as one LM-crew death (p_survive 0.5 across the exposed pair); "
            "the survivor and Collins return."
        ),
    },
    "surface_lm_electrical_failure": {
        "p_survive": 0.0,
        "outcome": "ascent_stage_cannot_arm",
        "explanation": (
            "An unrecoverable LM electrical/switchgear failure on the "
            "surface prevents arming the ascent engine. Sourced anchor: "
            "Apollo 11's own snapped ascent-engine arming breaker (closed "
            "with a felt-tip pen) — 1 serious incident in 6 landings, with "
            "~5% of such anomalies assumed beyond any workaround. The LM "
            "crew is stranded on the surface; Collins returns alone."
        ),
    },
    "sm_failure_translunar": {
        "p_survive": 0.90,
        "outcome": "lm_lifeboat_abort",
        "explanation": (
            "Catastrophic SM systems failure (Apollo-13 class: cryo tank / "
            "fuel cell) during the outbound coast with the LM attached. The "
            "LM-lifeboat abort is the demonstrated recovery — Apollo 13 "
            "survived exactly this — but with real margin risk (power, "
            "water, CO2 scrubbing, entry-corridor control). ~90% survival "
            "(1-for-1 historically, with documented thin margins). Sourced "
            "occurrence: 1 catastrophic SM failure in 15 crewed CSM flights."
        ),
    },
    "sm_failure_lunar_orbit": {
        "p_survive": 0.85,
        "outcome": "lm_lifeboat_abort_lunar",
        "explanation": (
            "Catastrophic SM failure in lunar orbit (pre-descent or "
            "post-ascent). LM lifeboat available for most of this window, "
            "but the abort starts from lunar orbit — TEI must be flown on "
            "DPS/APS with tighter consumables than Apollo 13's free-return "
            "geometry. Slightly worse than the translunar case (~85%)."
        ),
    },
    "sm_failure_surface": {
        "p_survive": 0.40,
        "outcome": "emergency_liftoff_to_dying_csm",
        "explanation": (
            "Catastrophic SM failure while the LM is on the surface: Collins "
            "is alone in the failing CSM; the LM must launch immediately "
            "(outside the planned liftoff window, degraded rendezvous "
            "geometry) and the combined stack then attempts a lifeboat "
            "return with a partially-expended LM. The worst pre-jettison "
            "geometry; ~40% (estimate — no historical analogue)."
        ),
    },
    "sm_failure_transearth": {
        "p_survive": 0.10,
        "outcome": "no_lifeboat_on_return",
        "explanation": (
            "Catastrophic SM failure during the trans-Earth coast — the LM "
            "was jettisoned before TEI; there is no lifeboat. It mostly kills "
            "all three: the CM's entry batteries and surge tanks last hours, "
            "not the days usually remaining to entry. Survival (~10%) is "
            "plausible only when the failure strikes late enough — within the "
            "last few hours before entry — that the crew can ride the "
            "remaining consumables down and return alive."
        ),
    },
    "transearth_no_entry": {
        "p_survive": 0.0,
        "outcome": "trapped_in_space",
        "explanation": (
            "Trans-Earth coast didn't reach Earth atmospheric entry interface. "
            "The spacecraft is on a trajectory that won't return to Earth "
            "before life support consumables (O2, CO2 scrubbers, power) are "
            "exhausted."
        ),
    },
    "entry_structural_failure_high_g": {
        "p_survive": 0.0,
        "outcome": "CM_breakup_in_atmosphere",
        "explanation": (
            "Atmospheric entry produced peak g-load above the Command "
            "Module's 12 g structural limit (the simulation's "
            "entry_structural_failure_high_g threshold; NASA TN D-6725 cites "
            "10 g as the guidance limit and ~12 g structural). Structural "
            "failure of the heat shield or pressure vessel leads to crew loss."
        ),
    },
    "entry_skip_out_or_breakup": {
        "p_survive": 0.0,
        "outcome": "lost_in_space_or_CM_breakup",
        "explanation": (
            "Entry flight path angle either too shallow (skip-out) — capsule "
            "bounces back into space on a long ballistic trajectory with no "
            "second-chance entry within consumables window — or too steep "
            "(burnup before parachute deployment)."
        ),
    },
    "synthetic_tei_failed": {
        "p_survive": 0.0,
        "outcome": "no_Earth_return_trajectory",
        "explanation": (
            "TEI targeting could not find a return trajectory. Stranded in "
            "lunar orbit until consumables exhausted."
        ),
    },
    "tei_dv_too_large": {
        "p_survive": 0.0,
        "outcome": "insufficient_propellant_for_return",
        "explanation": (
            "Required TEI ΔV exceeded SPS capability. No way home."
        ),
    },
    "tei_insufficient_propellant": {
        "p_survive": 0.0,
        "outcome": "insufficient_propellant_for_return",
        "explanation": (
            "The CSM doesn't have enough SPS propellant remaining at TEI "
            "time to execute the required burn. This usually means previous "
            "burns (LOI especially) consumed more propellant than nominal "
            "due to engine ISP/thrust dispersions. The crew is stranded in "
            "lunar orbit with no return path."
        ),
    },
}


def _estimate_csm_solo_return_rate(df):
    """Estimate P(Collins gets home | LM crew lost during descent/ascent).

    When the LM crew is lost, Collins is in a HEALTHY CSM that has already
    reached lunar orbit. His survival is then governed by the same TEI +
    trans-Earth + entry phases the CSM flies anyway — NOT an additional
    independent hazard. We therefore estimate it as the CONDITIONAL return
    rate given the CSM reached lunar orbit and actually flew those phases:

        P = successes / (successes + CSM-shared downstream failures)

    where CSM-shared downstream failures are tei_*, transearth_*, entry_*.
    This deliberately EXCLUDES the LM-phase failures themselves (they are the
    condition, not a hazard to Collins), so it cannot be dragged artificially
    low by descent losses. Falls back to a conservative value if data is thin.
    """
    if 'mission_failure' not in df.columns:
        return 0.95
    fails = df['mission_failure'].astype(str)
    successes = int(df['full_success'].fillna(False).sum())
    csm_downstream_loss = int(fails.str.startswith(('tei_', 'transearth_', 'entry_')).sum())
    denom = successes + csm_downstream_loss
    if denom < 20:
        return 0.95
    return float(successes / denom)


def apply_survival_model(df, seed=42):
    """Add PER-ASTRONAUT crew survival columns to the results dataframe.

    Produces, for each astronaut in CREW:
        survived_<name>        Boolean (random draw)
        psurv_<name>           survival probability used for that astronaut
    plus aggregate columns:
        crew_outcome           categorical label for the trial
        n_astronauts_survived  0..3 for the trial
    Backward-compatible columns crew_survived / crew_survival_prob are kept,
    defined as "all three survived" / "min individual probability".
    """
    rng = np.random.default_rng(seed)
    # Work on a contiguous 0..n-1 index: the draw loops mix iterrows()
    # (label-based) with positional numpy-array indexing, so a shuffled or
    # gapped index would misalign draws or raise. Production always passes a
    # fresh-from-CSV RangeIndex, but reset_index makes the function safe for
    # any filtered/concatenated frame.
    df = df.reset_index(drop=True)
    n = len(df)

    csm_solo_rate = _estimate_csm_solo_return_rate(df)

    # Per-astronaut probability and survival arrays
    psurv = {name: np.zeros(n) for name in CREW}
    surv  = {name: np.zeros(n, dtype=bool) for name in CREW}
    # PER-ASTRONAUT death cause: each astronaut who dies is attributed to what
    # actually killed THEM, not the trial's mission_failure label. A crewman
    # who survived the precipitating event but then died on the Earth-return
    # leg (e.g. Collins returning solo after the LM crew is lost, or a crew
    # that aborted successfully but lost the return) is charged to
    # RETURN_LEG_LOSS, not the upstream LM-phase cause.
    RETURN_LEG_LOSS = "crew_return_leg_loss"
    dcause = {name: [''] * n for name in CREW}
    outcome = ['nominal'] * n

    for i, row in df.iterrows():
        fail = row.get('mission_failure')
        nominal = pd.isna(fail) or fail is None
        if nominal:
            # A FULLY SUCCESSFUL mission means the crew splashed down and was
            # recovered — survival is 1.0 by definition (Apollo had zero
            # recovery fatalities across all crewed flights). All residual
            # crew risk lives in the explicit failure modes, not here. Using
            # 0.999 previously produced ~10 phantom crew losses on missions
            # flagged full_success — a self-contradiction the death-accounting
            # surfaced.
            p = 1.0
            for name in CREW:
                psurv[name][i] = p
            outcome[i] = 'nominal_splashdown'
            continue

        # Normalize failure-mode aliases to their SURVIVAL_MODEL key so new
        # or suffixed mode names don't silently fall through to the 0.5
        # default (which both mislabels the outcome and biases the aggregate).
        _fk = str(fail)
        if _fk not in SURVIVAL_MODEL:
            if _fk.startswith("missed_lunar_soi"):
                _fk = "missed_lunar_soi"            # post-MCC SOI miss == free-return-class
            elif _fk.startswith("descent_"):
                # Any other descent-phase loss (radar dropout, unrecovered AGC
                # alarm) is a degraded-touchdown abort: the ascent stage can
                # stage off, same class as the propellant-exhaustion abort.
                _fk = "descent_descent_propellant_exhausted"
        model = SURVIVAL_MODEL.get(_fk, None)
        if model is None:
            # Unmodeled failure: surface the gap loudly rather than silently
            # assigning 0.5 — every production mode should have an entry.
            print(f"  WARNING: no survival model for failure mode '{fail}' "
                  f"-> defaulting p_survive=0.5")
        p_mode = model['p_survive'] if model else 0.5
        outcome[i] = (model['outcome'] if model else 'unknown_abort')
        exposed = crew_exposed_for_failure(fail)

        if exposed == "all":
            # Whole stack shares the fate: a single survival probability for
            # all three (they are in the same vehicle for this phase).
            for name in CREW:
                psurv[name][i] = p_mode
        else:  # "lm_crew" — Armstrong & Aldrin at risk; Collins in the CSM
            for name in LM_CREW:
                psurv[name][i] = p_mode
            # Collins is alive in lunar orbit; his fate is the solo Earth
            # return. Tag the outcome to make the divergence explicit.
            for name in CSM_CREW:
                psurv[name][i] = csm_solo_rate
            outcome[i] = outcome[i] + "+CMP_solo_return"

    # Independent draws per astronaut. For "all" phases the three share a
    # single fate (same vehicle). For LM phases the logic is two-branch:
    #   - LM abort SUCCEEDS (prob = mode p_survive): Armstrong & Aldrin reach
    #     orbit, rendezvous with Collins, and ALL THREE fly home together,
    #     sharing one downstream Earth-return draw (csm_solo_rate). So each of
    #     the three survives iff (abort succeeded) AND (the shared return
    #     succeeded).
    #   - LM abort FAILS (prob = 1 - p_survive): Armstrong & Aldrin are lost;
    #     Collins returns solo and survives at csm_solo_rate.
    # This guarantees Collins is never worse off than the LM crew, as physics
    # requires (he faces a strict subset of their risks).
    u_shared = rng.uniform(0, 1, n)      # shared-fate draw (same vehicle)
    for i, row in df.iterrows():
        fail = row.get('mission_failure')
        nominal = pd.isna(fail) or fail is None
        exposed = "all" if nominal else crew_exposed_for_failure(fail)
        if exposed == "all":
            survived = u_shared[i] < psurv[CREW[0]][i]
            for name in CREW:
                surv[name][i] = survived
                if not survived:
                    dcause[name][i] = str(fail)   # shared-vehicle event
        elif exposed == "one_lm_crew":
            # Terminal EVA suit failure: ONE moonwalker (chosen at random)
            # dies of the suit failure; the partner aborts the EVA and, with
            # Collins, completes the ascent/rendezvous/return at the
            # Earth-return rate. The random victim choice is what makes
            # Armstrong and Aldrin differ (over many such trials, ~half each).
            victim = LM_CREW[int(rng.integers(0, 2))]
            partner = LM_CREW[1] if victim == LM_CREW[0] else LM_CREW[0]
            surv[victim][i] = False
            dcause[victim][i] = str(fail)         # the suit failure killed the victim
            surv[partner][i] = rng.uniform(0, 1) < csm_solo_rate
            surv["Collins"][i] = rng.uniform(0, 1) < csm_solo_rate
            if not surv[partner][i]:
                dcause[partner][i] = RETURN_LEG_LOSS
            if not surv["Collins"][i]:
                dcause["Collins"][i] = RETURN_LEG_LOSS
            psurv[victim][i] = 0.0
            psurv[partner][i] = csm_solo_rate
            psurv["Collins"][i] = csm_solo_rate
        else:
            p_abort = psurv[LM_CREW[0]][i]        # LM abort-stage success prob
            abort_ok = rng.uniform(0, 1) < p_abort
            return_ok = rng.uniform(0, 1) < csm_solo_rate   # shared return leg
            if abort_ok:
                # All three survived the LM event and reunited; they share the
                # return. If that return fails, all three die ON THE RETURN
                # LEG — not from the LM-phase cause they already survived.
                for name in CREW:
                    surv[name][i] = return_ok
                    if not return_ok:
                        dcause[name][i] = RETURN_LEG_LOSS
            else:
                # LM crew lost to the LM-phase event; Collins returns solo.
                for name in LM_CREW:
                    surv[name][i] = False
                    dcause[name][i] = str(fail)   # the LM event killed the moonwalkers
                surv["Collins"][i] = return_ok
                if not return_ok:
                    dcause["Collins"][i] = RETURN_LEG_LOSS  # solo-return loss, NOT the LM event
            # Record Collins's effective probability for reporting.
            psurv["Collins"][i] = csm_solo_rate

    df = df.copy()
    for name in CREW:
        df[f'psurv_{name}'] = psurv[name]
        df[f'survived_{name}'] = surv[name]
        df[f'dcause_{name}'] = dcause[name]   # per-astronaut death cause ('' if survived)
    df['n_astronauts_survived'] = sum(df[f'survived_{name}'].astype(int) for name in CREW)
    # Backward-compatible aggregate columns
    df['crew_survived'] = df['n_astronauts_survived'] == CREW_SIZE
    df['crew_survival_prob'] = np.minimum.reduce([psurv[name] for name in CREW])
    df['crew_outcome'] = outcome
    df.attrs['csm_solo_return_rate'] = csm_solo_rate
    return df


def generate_survival_report(df, outdir=OUTDIR):
    """Generate crew-survival analysis files (per-astronaut)."""
    n = len(df)
    _ms_col = 'mission_success' if 'mission_success' in df.columns else 'full_success'
    n_mission_success = int(df[_ms_col].fillna(False).astype(bool).sum())
    csm_solo = df.attrs.get('csm_solo_return_rate', float('nan'))

    # Per-astronaut survival counts
    per = {name: int(df[f'survived_{name}'].sum()) for name in CREW}

    lines = []
    lines.append("=" * 70)
    lines.append("APOLLO 11 — CREW SURVIVAL ANALYSIS (per astronaut)")
    lines.append("=" * 70)
    lines.append(f"Total missions simulated:           {n}")
    lines.append(f"Mission success (Moon + return):    {n_mission_success}  ({100*n_mission_success/n:.1f}%)")
    lines.append("")
    lines.append("PER-ASTRONAUT SURVIVAL")
    lines.append("-" * 70)
    lines.append(f"  {'Astronaut':<12}{'Role':<28}{'Survived':>12}{'Rate':>9}")
    roles = {"Armstrong": "CDR — Lunar Module (surface)",
             "Aldrin":    "LMP — Lunar Module (surface)",
             "Collins":   "CMP — Command Module (orbit)"}
    for name in CREW:
        lines.append(f"  {name:<12}{roles[name]:<28}{per[name]:>8}/{n:<3}{100*per[name]/n:>7.1f}%")
    lines.append("")
    # Distribution over number of survivors per mission
    dist = df['n_astronauts_survived'].value_counts().sort_index()
    lines.append("  Astronauts returned alive per mission:")
    for k in range(CREW_SIZE + 1):
        c = int(dist.get(k, 0))
        lines.append(f"     {k} of 3 survived: {c:>5} missions ({100*c/n:.1f}%)")
    total_astro = n * CREW_SIZE
    total_alive = sum(per.values())
    lines.append("")
    lines.append(f"  Total astronaut-flights:  {total_astro}")
    lines.append(f"  Returned alive:           {total_alive}  ({100*total_alive/total_astro:.1f}%)")
    lines.append(f"  Lost:                     {total_astro - total_alive}")
    lines.append("")
    lines.append("WHY THE NUMBERS DIFFER BY ASTRONAUT")
    lines.append("-" * 70)
    lines.append("Survival diverges by which vehicle each crewman occupies during")
    lines.append("the lunar phase. Collins (CMP) stays in the Command Module in")
    lines.append("lunar orbit and is NOT exposed to powered-descent, surface, or")
    lines.append("ascent risk. If the Lunar Module crew is lost, Collins's")
    lines.append("documented contingency was to return to Earth ALONE — so an LM")
    lines.append("loss does not kill him, though he must still complete a solo")
    lines.append(f"Earth return (modeled success rate {100*csm_solo:.1f}%, from this")
    lines.append("run's own TEI+entry statistics).")
    lines.append("")
    lines.append("Armstrong (CDR) and Aldrin (LMP) ride the Lunar Module to the")
    lines.append("surface and back, exposed to every risk Collins faces PLUS the")
    lines.append("descent/surface/ascent phases — so their survival is lower.")
    lines.append("")
    lines.append("NOTE: Armstrong and Aldrin share the same vehicle for every")
    lines.append("phase this simulation models, so their survival is identical")
    lines.append("here. Distinguishing them would require individual-level risks")
    lines.append("(EVA suit failure, a fall, a medical event) that are not")
    lines.append("modeled and for which no sourced rates exist. The meaningful,")
    lines.append("defensible distinction is Collins (CSM) vs the LM crew.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("OUTCOME BREAKDOWN — what happened to the crew, by failure mode")
    lines.append("=" * 70)
    lines.append("")

    # Group by failure mode
    if 'mission_failure' in df.columns:
        for fail_mode in df['mission_failure'].fillna('NOMINAL').unique():
            subset = df[df['mission_failure'].fillna('NOMINAL') == fail_mode]
            if fail_mode == 'NOMINAL':
                continue
            n_sub = len(subset)
            n_surv = int(subset['crew_survived'].sum())
            p_model = subset['crew_survival_prob'].iloc[0] if len(subset) > 0 else 0
            outcome = subset['crew_outcome'].iloc[0] if len(subset) > 0 else "unknown"
            expl = subset['crew_outcome'].iloc[0]
            model = SURVIVAL_MODEL.get(fail_mode, {})
            label = model.get('outcome', fail_mode)
            full_expl = model.get('explanation', '')
            lines.append(f"  {fail_mode}")
            lines.append(f"     Trials with this failure: {n_sub}")
            lines.append(f"     Recovery path: {label}")
            lines.append(f"     Survival probability: {p_model*100:.0f}%")
            lines.append(f"     Crew survived (sampled): {n_surv}/{n_sub} ({100*n_surv/max(1,n_sub):.0f}%)")
            if full_expl:
                import textwrap
                wrapped = textwrap.wrap(full_expl, width=66)
                lines.append("     Explanation:")
                for w in wrapped:
                    lines.append(f"       {w}")
            lines.append("")

    with open(os.path.join(outdir, "crew_survival.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ crew_survival.txt ({len(lines)} lines)")

    # Plot: ASTRONAUT DEATHS BY CAUSE (the crew-survival analogue of the
    # failure-mode breakdown). All three returned alive in most trials; this
    # figure shows what killed astronauts in the rest. Deaths per trial =
    # 3 - n_astronauts_survived, grouped by the trial's mission_failure cause.
    _all_three = int((df['n_astronauts_survived'] == CREW_SIZE).sum())
    deaths_per = (3 - df['n_astronauts_survived'])
    total_deaths = int(deaths_per.sum())
    # Group by PER-ASTRONAUT death cause (dcause_<name>), so a crewman who
    # died on the Earth-return leg after surviving the precipitating event is
    # charged to crew_return_leg_loss, not the upstream LM-phase mode.
    import pandas as _pd
    _causes = []
    for nm in CREW:
        dc = df.loc[~df[f'survived_{nm}'], f'dcause_{nm}']
        _causes.extend([c if c else '(unattributed)' for c in dc])
    by_cause = _pd.Series(_causes).value_counts()
    by_cause = by_cause[by_cause > 0]
    # Color by mission phase (prefix), self-contained. High-contrast categorical
    # palette (matches generate_outputs' phase_colors) so the cause bars are
    # easy to tell apart rather than a ramp of blues and purples.
    def _phase_color(c):
        c = str(c)
        if c.startswith("launch"): return "#1f77b4"          # blue
        if c.startswith("sm_failure_translunar") or c.startswith("missed_lunar") or c.startswith("transposition"): return "#ff7f0e"  # orange
        if c.startswith("sm_failure_lunar") or c.startswith("sm_failure_surface"): return "#2ca02c"  # green
        if c.startswith("descent"): return "#d62728"         # red
        if c.startswith("surface"): return "#9467bd"         # purple
        if c.startswith("ascent"): return "#8c564b"          # brown
        if c.startswith("rendezvous"): return "#e377c2"      # pink
        if c.startswith("tei"): return "#17becf"             # cyan
        if c.startswith("sm_failure_transearth") or c.startswith("transearth"): return "#bcbd22"  # olive
        if c.startswith("entry"): return "#393b79"           # navy
        return "#7f7f7f"                                     # gray (e.g. return-leg loss)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    y_pos = np.arange(len(by_cause))
    colors = [_phase_color(c) for c in by_cause.index]
    bars = ax.barh(y_pos, by_cause.values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    def _pretty(c):
        # Readable label from the raw mission_failure token: collapse the
        # doubled "descent_descent" prefix (the code emits "descent_"+reason
        # where reason already starts with "descent_") and de-underscore.
        s = str(c).replace("descent_descent_", "descent_")
        return s.replace("_", " ")
    ax.set_yticklabels([_pretty(c) for c in by_cause.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Astronaut deaths (count)")
    ax.set_title(f"Astronaut deaths by cause "
                 f"({total_deaths} of {3*n} astronaut-missions, "
                 f"{100*total_deaths/(3*n):.1f}%)")
    for b, c in zip(bars, by_cause.values):
        ax.text(c + total_deaths*0.005, b.get_y() + b.get_height()/2,
                f"{int(c)} ({100*c/total_deaths:.0f}%)", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "crew_survival.png"), dpi=110)
    plt.close()
    print(f"  ✓ crew_survival.png")

    # Pie chart: mission vs crew outcomes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    # Mission success (landed on the Moon AND all three returned alive)
    _ms_col = 'mission_success' if 'mission_success' in df.columns else 'full_success'
    mission_succ = int(df[_ms_col].fillna(False).astype(bool).sum())
    mission_fail = n - mission_succ
    ax1.pie([mission_succ, mission_fail],
             labels=[f'Mission success\n({mission_succ})',
                      f'Mission failure\n({mission_fail})'],
             colors=['#2D5A3D', '#C73E1D'], autopct='%1.1f%%',
             startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax1.set_title('Mission Success Rate')
    # Per-astronaut survival bar chart (replaces the single crew pie)
    per = {name: int(df[f'survived_{name}'].sum()) for name in CREW}
    names = list(CREW)
    rates = [100*per[nm]/n for nm in names]
    bar_colors = ['#C73E1D', '#C73E1D', '#2D5A3D']  # LM crew vs CMP
    bars2 = ax2.bar(names, rates, color=bar_colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Survival rate (%)')
    # Zoom the y-axis so the (small) inter-astronaut differences are visible —
    # the survival rates all sit in the low-to-mid 90s, so a 0-100 axis hides
    # the Collins-vs-moonwalker gap entirely.
    _ymin = min(90.0, min(rates) - 1.0)
    ax2.set_ylim(_ymin, 100)
    ax2.set_title(f'Survival by astronaut (y-axis from {_ymin:.0f}%)')
    for b, nm in zip(bars2, names):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                 f"{per[nm]}/{n}\n{100*per[nm]/n:.1f}%", ha='center',
                 va='bottom', fontsize=9)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor='#C73E1D', label='LM crew (surface)'),
                        Patch(facecolor='#2D5A3D', label='CMP (orbit)')],
               loc='lower center', fontsize=8)
    fig.suptitle(f"Apollo 11 Monte Carlo — Mission vs Crew Outcomes ({n} trials)",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mission_vs_crew.png"), dpi=110)
    plt.close()
    print(f"  ✓ mission_vs_crew.png")


def main():
    csv_path = os.path.join(OUTDIR, "results.csv")
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} trials loaded")

    print("Applying crew survival model...")
    df = apply_survival_model(df)

    # MISSION SUCCESS — the headline metric. A trial is a success if the crew
    # LANDED on the Moon AND all three returned to Earth alive, EVEN IF an
    # in-flight anomaly occurred that the crew recovered from (e.g. a docking
    # failure resolved by a contingency EVA crew transfer, or a service-module
    # failure that struck late enough on the trans-Earth coast for the crew to
    # ride the remaining consumables home). This is NASA's actual objective —
    # "land a man on the Moon and return him safely to the Earth" — and it is a
    # strict superset of `full_success`, the simulator's flag for a FLAWLESS
    # mission with no recorded anomaly at all. "Landed" is the real touchdown
    # (`land_lat_deg` recorded), not the `descent_success` flag, which defaults
    # True for trials that abort before descent ever runs.
    _landed = (df['land_lat_deg'].notna() if 'land_lat_deg' in df.columns
               else df['full_success'].fillna(False).astype(bool))
    df['mission_success'] = (_landed & df['crew_survived'].fillna(False).astype(bool))

    # Save augmented CSV
    out_csv = os.path.join(OUTDIR, "results_with_survival.csv")
    df.to_csv(out_csv, index=False)
    print(f"  ✓ {out_csv}")

    print("Generating survival report and plots...")
    generate_survival_report(df)
    print("Done.")


if __name__ == "__main__":
    main()
