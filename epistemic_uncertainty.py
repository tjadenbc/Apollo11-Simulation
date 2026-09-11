#!/usr/bin/env python3
"""Parameter (epistemic) uncertainty of the headline estimate.

The Wilson interval quoted with the mission-success rate measures only the
Monte Carlo sampling error of the 10,000-trial estimate AT the input failure
rates. This script quantifies the other, much larger, source of uncertainty:
the input rates themselves, where they rest on a countable but tiny record --

    service-module catastrophic failure   1 event  in 15 crewed CSM flights
    S-IVB restart failure (TLI)           0 events in 10 post-fix restarts
    docking capture anomaly              ~2 events in ~21 program dockings

Each of these enters the simulation as an INDEPENDENT Bernoulli draw per trial
(the SM occurrence and both docking draws are pre-generated in
``sample_perturbation``; the restart is a phase-local salted draw), so the
fleet's outcome at any other rate p' is an importance reweighting, exact in form, of the
existing trials: a trial that drew the event carries weight p'/p-hat, one that
did not carries (1-p')/(1-p-hat), where p-hat is the realized draw fraction.
For a single mode this reduces to the post-stratified identity

    P(success | p') = (1 - p') * S0 + p' * S1

with S0/S1 the success rates among the trials that did not / did draw the
event -- exact in form and linear in p', adding no Monte Carlo noise of its own
beyond the fleet's, which it inherits through S0 and S1 (the script reports that
standard error next to each sensitivity row: it grows as p' RISES, the struck
stratum being the sparse one -- except where that stratum has no variance at all,
as for the restart, whose failed trials contain no success, and there it shrinks).
Propagating a Beta posterior on each rate through the identity therefore gives
the same posterior distribution of the success probability that an outer sampling
loop over the rates would produce, without re-running anything.

The draws are regenerated from the fleet's seed and checked against the
results file (every trial whose outcome shows the event must have drawn it);
a wrong seed or feature-flag set fails that check loudly.

Usage:
    python3 epistemic_uncertainty.py [outdir] [seed]      # default outputs/final 37

Writes <outdir>/epistemic_uncertainty.txt and .json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs", "final")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 37

# The paper's input rates and the records behind them.
SM_K, SM_N = 1, 15              # Apollo 13 in 15 crewed CSM flights
RS_K, RS_N = 0, 10              # post-Apollo-6 J-2 restarts, zero failures
DK_K, DK_N = 2, 21              # capture anomalies in program dockings
P_SM = 1.0 / 15.0
P_RS = 4.5e-2                   # Jeffreys posterior mean on 0 of 10
P_DK = 0.0095                   # 2/21 x unrecovered fraction
DK_UNRECOVERED = P_DK / (DK_K / DK_N)   # the unrecovered fraction the paper applies

N_POST = 200_000                # posterior draws for the credible intervals
POST_SEED = 2027


def load_fleet(outdir, seed):
    """Regenerate the per-trial Bernoulli draws and align them with the results."""
    os.environ.setdefault("APOLLO_SKIP_NOMINAL_CHECK", "1")
    sys.path.insert(0, HERE)
    import apollo11 as A
    cfg_path = os.path.join(outdir, "run_config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if k.startswith("ENABLE_") and hasattr(A, k):
                setattr(A, k, v)          # the run's feature flags gate some draws
    d = pd.read_csv(os.path.join(outdir, "results_with_survival.csv"), low_memory=False)
    d = d.sort_values("trial").reset_index(drop=True)
    n = len(d)
    assert (d.trial.to_numpy() == np.arange(n)).all(), "trial index has gaps"
    rng = np.random.default_rng(seed)
    P = [A.sample_perturbation(rng) for _ in range(n)]
    lab = d.mission_failure.fillna("")
    F = dict(
        ms=d.mission_success.fillna(False).astype(bool).to_numpy(),
        cr=d.crew_survived.fillna(False).astype(bool).to_numpy(),
        ld=d.land_lat_deg.notna().to_numpy(),          # reached the lunar surface (paper Sec. 7.1)
        sm=np.array([bool(p["sm_failure"]) for p in P]),
        td=np.array([bool(p["td_docking_failed"]) for p in P]),
        rdv=np.array([bool(p["docking_failed"]) for p in P]),
        reached=~lab.str.startswith("launch_").to_numpy(),   # reached the TLI restart
        rfail=(lab == "tli_s_ivb_restart_failure").to_numpy(),
    )
    # --- consistency checks: every manifested event must have been drawn -----
    shows_sm = lab.str.startswith("sm_failure").to_numpy()
    bad = int((shows_sm & ~F["sm"]).sum())
    assert bad == 0, f"{bad} trials show an SM failure the regenerated stream did not draw -- wrong seed/flags?"
    assert int(((lab == "transposition_docking_failure").to_numpy() & ~F["td"]).sum()) == 0
    assert int(((lab == "rendezvous_docking_failure").to_numpy() & ~F["rdv"]).sum()) == 0
    assert F["rfail"][~F["reached"]].sum() == 0
    F["checks"] = dict(sm_shown=int(shows_sm.sum()), sm_drawn=int(F["sm"].sum()),
                       sm_drawn_unmanifested=int((F["sm"] & ~shows_sm).sum()),
                       td_drawn=int(F["td"].sum()), rdv_drawn=int(F["rdv"].sum()),
                       reached_tli=int(F["reached"].sum()), restart_failed=int(F["rfail"].sum()))
    return F


def build_strata(F):
    """Collapse the fleet to its 2x3x2x2 draw strata (exact; weights are constant within one)."""
    rs_state = np.where(~F["reached"], 0, np.where(F["rfail"], 2, 1))
    key = F["sm"].astype(int) * 12 + rs_state * 4 + F["td"].astype(int) * 2 + F["rdv"].astype(int)
    n_k = np.bincount(key, minlength=24).astype(float)
    s_k = np.bincount(key, weights=F["ms"], minlength=24)
    c_k = np.bincount(key, weights=F["cr"], minlength=24)
    l_k = np.bincount(key, weights=F["ld"], minlength=24)
    phat = dict(sm=F["sm"].mean(), td=F["td"].mean(), rdv=F["rdv"].mean(),
                rs=F["rfail"].sum() / F["reached"].sum())
    return n_k, s_k, c_k, l_k, phat


def weights(k, p_sm, p_rs, p_dk, phat, p_dk2=None):
    s, r, t, v = k // 12, (k // 4) % 3, (k // 2) % 2, k % 2
    p_dk2 = p_dk if p_dk2 is None else p_dk2   # both dockings share a rate except in the self-check
    w = np.where(s, p_sm / phat["sm"], (1 - p_sm) / (1 - phat["sm"]))
    w = w * np.where(t, p_dk / phat["td"], (1 - p_dk) / (1 - phat["td"]))
    w = w * np.where(v, p_dk2 / phat["rdv"], (1 - p_dk2) / (1 - phat["rdv"]))
    w = w * np.where(r == 0, 1.0, np.where(r == 2, p_rs / phat["rs"], (1 - p_rs) / (1 - phat["rs"])))
    return w


def estimate(strata, p_sm, p_rs, p_dk, p_dk2=None):
    """Mission / crew success (percent) at the given rates; vectorised over arrays of rates."""
    n_k, s_k, c_k, l_k, phat = strata
    p_sm, p_rs, p_dk = np.broadcast_arrays(np.atleast_1d(p_sm), np.atleast_1d(p_rs), np.atleast_1d(p_dk))
    if p_dk2 is not None:
        p_dk2 = np.broadcast_to(np.atleast_1d(p_dk2), p_dk.shape)
    W = np.stack([weights(k, p_sm, p_rs, p_dk, phat, p_dk2) for k in range(24)], axis=-1)
    den = W @ n_k
    return 100 * (W @ s_k) / den, 100 * (W @ c_k) / den, 100 * (W @ l_k) / den


def main():
    F = load_fleet(OUTDIR, SEED)
    strata = build_strata(F)
    n_k, s_k, c_k, l_k, phat = strata
    n = int(n_k.sum())
    out, J = [], {}
    P = lambda s: out.append(s)

    fleet_m, fleet_c = 100 * F["ms"].mean(), 100 * F["cr"].mean()
    se = 100 * np.sqrt(F["ms"].mean() * (1 - F["ms"].mean()) / n)
    # Name the run directory RELATIVE to this file: an absolute path here would bake the
    # author's local filesystem layout into a released data product.
    _rel = os.path.relpath(OUTDIR, HERE)
    P(f"EPISTEMIC (PARAMETER) UNCERTAINTY OF THE HEADLINE -- {_rel}  (n={n}, seed {SEED})")
    P("")
    P("Draw-stream check (regenerated from the seed, compared with the results file):")
    P(f"  SM failure: {F['checks']['sm_shown']} trials show one, all {F['checks']['sm_shown']} were drawn; "
      f"{F['checks']['sm_drawn']} drawn in all. The other {F['checks']['sm_drawn_unmanifested']} were drawn in trials "
      f"that had already ended -- on an earlier failure, or on a recovered anomaly closing the flight -- before the strike time.")
    P(f"  dockings drawn: T&D {F['checks']['td_drawn']}, rendezvous {F['checks']['rdv_drawn']};  "
      f"reached TLI restart {F['checks']['reached_tli']}, restart failed {F['checks']['restart_failed']}")
    P(f"  realized draw fractions: SM {100*phat['sm']:.2f}%  restart {100*phat['rs']:.2f}%  "
      f"T&D {100*phat['td']:.2f}%  rendezvous {100*phat['rdv']:.2f}%")
    m0, c0, _ = estimate(strata, phat["sm"], phat["rs"], phat["td"], phat["rdv"])
    P(f"  identity reconstructs the fleet at its realized fractions: mission {m0[0]:.2f}% vs {fleet_m:.2f}%, "
      f"crew {c0[0]:.2f}% vs {fleet_c:.2f}%")
    P("")
    P(f"Monte Carlo sampling uncertainty (for scale): fleet {fleet_m:.2f}% mission success, "
      f"binomial s.e. {se:.2f} pt, Wilson 95% half-width ~{1.96*se:.2f} pt")
    J["fleet"] = dict(mission=fleet_m, crew=fleet_c, mc_se_pt=se, n=n)
    J["realized_fractions"] = phat

    def se_pt(y, drawn, p):
        """Monte Carlo s.e. (percentage points) of the post-stratified estimate at rate p.
        The stratum weight totals are fixed by p, so the variance is the within-stratum
        binomial one, (1-p)^2 S0(1-S0)/n0 + p^2 S1(1-S1)/n1 -- NOT the delta-method variance
        of a random-weight importance estimator. Checked against a 4,000-resample bootstrap."""
        n0, n1 = int((~drawn).sum()), int(drawn.sum())
        a, b = y[~drawn].mean(), y[drawn].mean()
        return 100 * np.sqrt((1 - p) ** 2 * a * (1 - a) / n0 + p ** 2 * b * (1 - b) / n1)

    # --- single-mode post-stratified rates ------------------------------------
    S0, S1 = 100 * F["ms"][~F["sm"]].mean(), 100 * F["ms"][F["sm"]].mean()
    C0, C1 = 100 * F["cr"][~F["sm"]].mean(), 100 * F["cr"][F["sm"]].mean()
    P("")
    P("Post-stratified success rates on the SM draw (the identity's S0, S1):")
    P(f"  not drawn: mission {S0:.2f}%  crew {C0:.2f}%      drawn: mission {S1:.2f}%  crew {C1:.2f}%")
    P(f"  slope: each +1 pt of SM occurrence rate costs {(S0-S1)/100:.2f} pt of mission success and {(C0-C1)/100:.2f} pt of crew survival")
    J["sm_strata"] = dict(S0=S0, S1=S1, C0=C0, C1=C1)

    # --- SM occurrence-rate sensitivity (others at the paper's rates) ---------
    cp_lo, cp_hi = beta.ppf(0.025, SM_K, SM_N - SM_K + 1), beta.ppf(0.975, SM_K + 1, SM_N - SM_K)
    jf = beta(SM_K + 0.5, SM_N - SM_K + 0.5)
    rows = [("Clopper-Pearson 95% lower bound", cp_lo),
            ("Jeffreys posterior 2.5%", jf.ppf(0.025)),
            ("frequentist point estimate 1/15 (paper)", P_SM),
            ("Jeffreys posterior median", jf.ppf(0.5)),
            ("Jeffreys posterior mean", jf.mean()),
            ("referee's illustrative 15%", 0.15),
            ("Jeffreys posterior 97.5%", jf.ppf(0.975)),
            ("Clopper-Pearson 95% upper bound", cp_hi)]
    P("")
    P("SM occurrence-rate sensitivity (S-IVB restart and docking at the paper's rates):")
    P(f"  {'basis':42} {'p_SM':>7}  {'mission':>8}  {'crew':>7}   (Monte Carlo s.e., pt)")
    J["sm_sensitivity"] = []
    for lab_, p in rows:
        m, c, _ = estimate(strata, p, P_RS, P_DK)
        sem, sec = se_pt(F["ms"], F["sm"], p), se_pt(F["cr"], F["sm"], p)
        P(f"  {lab_:42} {100*p:6.2f}%  {m[0]:7.1f}%  {c[0]:6.1f}%   +/-{sem:.2f} / {sec:.2f}")
        J["sm_sensitivity"].append(dict(basis=lab_, p_sm=p, mission=m[0], crew=c[0], se_mission=sem, se_crew=sec))

    def se_rs(y, p):
        """s.e. of the whole-fleet estimate when only the restart rate moves. Three strata:
        trials that never reached the draw (weight fixed at the realized reached fraction f),
        and the reached trials split by outcome."""
        nev, rch, fl = ~F["reached"], F["reached"], F["rfail"]
        f = rch.mean()
        a, n_a = y[nev].mean(), int(nev.sum())
        b, n_b = y[rch & ~fl].mean(), int((rch & ~fl).sum())
        c_, n_c = y[rch & fl].mean(), int((rch & fl).sum())
        v = ((1 - f) ** 2) * a * (1 - a) / n_a + (f ** 2) * (
            (1 - p) ** 2 * b * (1 - b) / n_b + p ** 2 * c_ * (1 - c_) / n_c)
        return 100 * np.sqrt(v)

    # --- S-IVB restart sensitivity ---------------------------------------------
    jr = beta(RS_K + 0.5, RS_N - RS_K + 0.5)
    P("")
    P("S-IVB restart-rate sensitivity (SM and docking at the paper's rates):")
    P(f"  {'basis':42} {'p_rs':>7}  {'mission':>8}  {'crew':>7}   (Monte Carlo s.e., pt)")
    J["restart_sensitivity"] = []
    for lab_, p in [("Jeffreys posterior 2.5%", jr.ppf(0.025)), ("Jeffreys posterior median", jr.ppf(0.5)),
                    ("Jeffreys posterior mean 4.55%, applied as 4.5% (paper)", P_RS), ("Jeffreys posterior 97.5%", jr.ppf(0.975)),
                    ("rule-of-three upper bound 3/10", 0.30)]:
        m, c, _ = estimate(strata, P_SM, p, P_DK)
        sem, sec = se_rs(F["ms"], p), se_rs(F["cr"], p)
        P(f"  {lab_:42} {100*p:6.2f}%  {m[0]:7.1f}%  {c[0]:6.1f}%   +/-{sem:.2f} / {sec:.2f}")
        J["restart_sensitivity"].append(dict(basis=lab_, p_rs=p, mission=m[0], crew=c[0],
                                             se_mission=sem, se_crew=sec))

    # --- joint posterior propagation --------------------------------------------
    R = np.random.default_rng(POST_SEED)
    P("")
    J["joint"] = {}
    for tag, a0 in (("Jeffreys Beta(1/2,1/2)", 0.5), ("uniform Beta(1,1)", 1.0)):
        p_sm = R.beta(SM_K + a0, SM_N - SM_K + a0, N_POST)
        p_rs = R.beta(RS_K + a0, RS_N - RS_K + a0, N_POST)
        p_dk = DK_UNRECOVERED * R.beta(DK_K + a0, DK_N - DK_K + a0, N_POST)
        m, c, ld = estimate(strata, p_sm, p_rs, p_dk)
        qm, qc = np.percentile(m, [2.5, 50, 97.5]), np.percentile(c, [2.5, 50, 97.5])
        ql = np.percentile(ld, [2.5, 50, 97.5])
        means = ((SM_K + a0) / (SM_N + 2 * a0), (RS_K + a0) / (RS_N + 2 * a0), DK_UNRECOVERED * (DK_K + a0) / (DK_N + 2 * a0))
        mm, cm, _ = estimate(strata, *means)
        P(f"Joint posterior over (p_SM, p_restart, p_dock) -- {tag} priors, {N_POST} draws:")
        P(f"  posterior-mean rates: SM {100*means[0]:.2f}%  restart {100*means[1]:.2f}%  docking {100*means[2]:.3f}%  "
          f"-> mission {mm[0]:.1f}%  crew {cm[0]:.1f}%")
        P(f"  mission success: mean {m.mean():.1f}%  median {qm[1]:.1f}%  95% credible interval [{qm[0]:.1f}, {qm[2]:.1f}]%  (width {qm[2]-qm[0]:.1f} pt)")
        P(f"  crew survival:   mean {c.mean():.1f}%  median {qc[1]:.1f}%  95% credible interval [{qc[0]:.1f}, {qc[2]:.1f}]%  (width {qc[2]-qc[0]:.1f} pt)")
        P(f"  landing rate:    mean {ld.mean():.1f}%  median {ql[1]:.1f}%  95% credible interval [{ql[0]:.1f}, {ql[2]:.1f}]%   (fleet {100*F['ld'].mean():.1f}%; paper Sec. 7.1)")
        share = {}
        for nm, arrs in (("SM", (p_sm, means[1], means[2])), ("restart", (means[0], p_rs, means[2])),
                         ("docking", (means[0], means[1], p_dk))):
            m1, c1, _ = estimate(strata, *arrs)
            share[nm] = dict(mission_ci=[float(np.percentile(m1, 2.5)), float(np.percentile(m1, 97.5))],
                             crew_ci=[float(np.percentile(c1, 2.5)), float(np.percentile(c1, 97.5))],
                             mission_sd=float(m1.std()), var_share=float(m1.var() / m.var()))
            P(f"    {nm:8} alone: mission 95% CrI [{share[nm]['mission_ci'][0]:.1f}, {share[nm]['mission_ci'][1]:.1f}]%  "
              f"crew [{share[nm]['crew_ci'][0]:.1f}, {share[nm]['crew_ci'][1]:.1f}]%  "
              f"sd {share[nm]['mission_sd']:.2f} pt  ({100*share[nm]['var_share']:.0f}% of the joint mission variance)")
        J["joint"][tag] = dict(means=dict(sm=means[0], restart=means[1], docking=means[2]),
                               mission_at_means=mm[0], crew_at_means=cm[0],
                               mission=dict(mean=m.mean(), median=qm[1], ci=[qm[0], qm[2]]),
                               crew=dict(mean=c.mean(), median=qc[1], ci=[qc[0], qc[2]]),
                               landing=dict(mean=ld.mean(), median=ql[1], ci=[ql[0], ql[2]]),
                               one_at_a_time=share)
        P("")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(OUTDIR, "epistemic_uncertainty.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUTDIR, "epistemic_uncertainty.json"), "w") as f:
        json.dump(J, f, indent=1, default=float)


if __name__ == "__main__":
    main()
