#!/usr/bin/env python3
"""Move this user's waiting jobs onto idle compute nodes (sim repo, 2026-10-29).

  queue_watcher.py run DIR [--interval S] [--recheck S] [--prefix P] [--names N[,N...]]
                   [--once] [--dry-run]

On this cluster idle gpu nodes take waiting jobs but idle compute nodes do not: a job
stays PENDING (Resources) while compute nodes sit idle, and pinning the pending job in
place (scontrol update ReqNodeList=) only turns it BadConstraints. A fresh
sbatch --nodelist does start. Unlike pin_watcher.py (one submit_mc.sh chain), this
watches every job of the user whose name starts with one of NAMES (default a11-). The
account is shared with other sessions (e.g. the A2 solver's a2-arbtab3 arrays), whose jobs
it must not move (2026-10-30). Each cycle, when whole idle nodes named PREFIX* exist and
such a job waits (Resources / Priority / BadConstraints / None), the oldest waiting job is
re-submitted pinned to an idle node and the original is cancelled:
  - same batch script (scontrol write batch_script), name, time limit, CPUs, partition,
    account, work dir and output paths; an array task keeps its index (--array=<i>),
    so $SLURM_ARRAY_TASK_ID is unchanged;
  - the submit-time environment is read from a running task of the same array
    (/proc/<batch pid>/environ through srun --overlap), minus what SLURM sets per node.
    With no running task to read, only a self-contained submit_mc.sh job (its env
    prefix inline in the --wrap) moves, under the watcher's environment. Anything else
    is left alone: a job configured through exported variables (e.g. AR2_*) and
    re-submitted without them would fly the wrong configuration;
  - every pending job whose dependency names the moved job, or its whole array, is
    re-pointed at the replacement plus the array's other unfinished tasks.
The original is held first and cancelled only after the replacement is queued and every
dependent rewired; a failed step cancels the replacement, restores the dependencies and
releases the original. Two snapshots RECHECK s apart must agree, so SLURM's own scheduler
gets first claim. Jobs already pinned (ReqNodeList set), multi-node jobs, jobs asking
for GRES or features and jobs submitted with script arguments (Command= holds more than
the script path) are never touched: scontrol write batch_script does not keep the
arguments, so the replacement would run without them (2026-10-30: 8 a2-arbtab3 tasks
failed at 0 s, "$1: unbound variable"). Each move is appended to DIR/moves.jsonl
(old id -> new id); the scripts sent are kept in DIR/scripts/. Runs until stopped.
"""
import json
import os
import re
import subprocess
import sys
import time

WAITING = {"Resources", "Priority", "BadConstraints", "None"}
NODE_VARS = re.compile(r"^(SLURM_|SLURMD_|SBATCH_|SALLOC_|SRUN_|CUDA_VISIBLE_DEVICES$|"
                       r"GPU_DEVICE_ORDINAL$|ROCR_VISIBLE_DEVICES$|ENVIRONMENT$|HOSTNAME$)")
DEP = re.compile(r"^(after|afterany|afterok|afternotok):([^()]+)\((\w+)\)$")
NULL = ("", "(null)", "N/A")
NEW = "@NEW@"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def snapshot(prefix):
    """This user's jobs (one row per array task) and the whole idle PREFIX* nodes."""
    rc, out, _ = sh(["squeue", "-h", "-r", "-u", os.environ["USER"],
                     "-o", "%i|%A|%F|%K|%T|%r|%V|%E|%j"])
    jobs = []
    for line in out.splitlines():
        f = line.split("|", 8)
        if len(f) == 9:
            jobs.append(dict(key=f[0], uid=f[1], array=f[2], task=f[3], state=f[4],
                             reason=f[5], submit=f[6], dep=f[7], name=f[8]))
    rc2, out, _ = sh(["sinfo", "-h", "-N", "-t", "idle", "-o", "%N|%C"])
    idle = set()
    for line in out.splitlines():
        node, _, cpus = line.partition("|")
        if node.startswith(prefix) and cpus.split("/")[0] == "0":
            idle.add(node)
    if rc or rc2:
        return None, []
    return jobs, sorted(idle)


def waiting(jobs, names):
    w = [j for j in jobs if j["state"] == "PENDING" and j["reason"] in WAITING
         and j["name"].startswith(names)]
    return sorted(w, key=lambda j: (j["submit"], j["array"],
                                    int(j["task"]) if j["task"].isdigit() else -1))


def details(key):
    _, out, _ = sh(["scontrol", "show", "job", "-dd", key])
    d = {}
    for k, v in re.findall(r"(\S+?)=(\S*)", out):
        d.setdefault(k, v)
    m = re.search(r"^\s*Command=(.*)$", out, re.M)   # the whole line: script plus its arguments
    if m:
        d["Command"] = m.group(1).strip()
    return d


def unsupported(d):
    if d.get("JobState") != "PENDING":
        return f"state {d.get('JobState')}"
    if d.get("BatchFlag") != "1":
        return "not a batch job"
    if d.get("NumNodes") not in ("1", "1-1"):
        return f"{d.get('NumNodes')} nodes"
    if d.get("ReqNodeList") not in NULL:
        return f"already pinned to {d['ReqNodeList']}"
    if d.get("Command") not in NULL and len(d["Command"].split()) > 1:
        return f"submitted with script arguments ({d['Command']})"
    if d.get("Features") not in NULL:
        return f"features {d['Features']}"
    for k, v in d.items():
        if k.startswith("TresPer") and v not in NULL:
            return f"{k}={v}"
    return None


def read_env(uid):
    """Submit-time environment of running job UID, from its batch process."""
    r = subprocess.run(
        ["timeout", "120", "srun", f"--jobid={uid}", "--overlap", "-N1", "-n1", "--quiet",
         "bash", "-c", f'p=$(pgrep -u "$USER" -f "/job0*{uid}/slurm_script" | head -1); '
                       f'[ -n "$p" ] && cat /proc/$p/environ'], capture_output=True)
    env = {}
    for e in r.stdout.split(b"\0"):
        k, eq, v = e.partition(b"=")
        if eq and k:
            k = k.decode("utf-8", "surrogateescape")
            if not NODE_VARS.match(k):
                env[k] = v.decode("utf-8", "surrogateescape")
    return env if "PATH" in env and "HOME" in env else None


def own_env():
    return {k: v for k, v in os.environ.items() if not NODE_VARS.match(k)}


def finished_tasks(array, live):
    """{task key: state} for tasks of ARRAY that are no longer queued."""
    _, out, _ = sh(["sacct", "-n", "-X", "-P", "-j", array, "-o", "JobID,State"])
    done = {}
    for line in out.splitlines():
        k, _, s = line.partition("|")
        if "[" not in k and k not in live and k.startswith(array + "_"):
            done[k] = s.split()[0] if s else ""
    return done


def acct_state(jid):
    if jid.endswith("_*"):          # a whole (e.g. replacement single-task) array
        s = set(finished_tasks(jid[:-2], set()).values())
        return None if not s else "COMPLETED" if s == {"COMPLETED"} else \
            sorted(s - {"COMPLETED"})[0]
    _, out, _ = sh(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "JobID,State"])
    for line in out.splitlines():
        k, _, s = line.partition("|")
        if k == jid:
            return s.split()[0] if s else ""
    return None


def rewrite(dep, job, jobs):
    """("ok", new dependency with NEW for the replacement), ("no", None) if DEP does not
    name JOB, ("doomed", why) if it can never be satisfied, ("bad", why) if not parsed."""
    arr, task = job["array"], job["task"]
    refs = {job["key"], job["uid"]} | ({arr} if task == "N/A" else set())
    queued = {j["key"] for j in jobs} | {j["uid"] for j in jobs}
    out, named = [], False
    for tok in dep.split(","):
        if tok.startswith("singleton"):
            out.append("singleton")
            continue
        m = DEP.match(tok)
        if not m:
            return "bad", f"dependency token {tok!r}"
        typ, jid, status = m.groups()
        if task != "N/A" and jid == f"{arr}_*":
            if typ not in ("afterok", "afterany"):
                return "bad", f"{typ} on a whole array"
            named = True
            live = [j["key"] for j in jobs if j["array"] == arr and j["task"] != "N/A"]
            if typ == "afterok":
                bad = {k: s for k, s in finished_tasks(arr, set(live)).items() if s != "COMPLETED"}
                if bad:
                    return "doomed", f"array {arr} has finished tasks {bad}"
            out += [f"{typ}:{NEW if k == job['key'] else k}" for k in live]
        elif jid in refs:
            named = True
            out.append(f"{typ}:{NEW}")
        elif status == "fulfilled":
            continue
        elif status == "failed":
            return "doomed", f"{tok} failed"
        elif jid in queued or (jid.endswith("_*") and any(j["array"] == jid[:-2] for j in jobs)):
            out.append(f"{typ}:{jid}")
        else:
            # gone from the queue: re-stating it would be refused once slurmctld purges it
            s = acct_state(jid)
            if s == "COMPLETED" or (s and typ in ("after", "afterany")):
                continue
            if s and typ == "afterok":
                return "doomed", f"{jid} ended {s}"
            out.append(f"{typ}:{jid}")
    if not named:
        return "no", None
    seen = []
    for t in out:
        if t not in seen:
            seen.append(t)
    return "ok", ",".join(seen)


def plan(job, jobs, wdir, envs):
    """Everything needed to move JOB, or (None, why)."""
    d = details(job["key"])
    why = unsupported(d)
    if why:
        return None, why
    rc, script, err = sh(["scontrol", "write", "batch_script", job["key"], "-"])
    if rc or not script.startswith("#!"):
        return None, f"batch script unreadable ({err or script[:60]})"
    env, src = None, None
    if job["task"] != "N/A":
        if job["array"] not in envs:
            sib = next((j for j in jobs if j["array"] == job["array"] and j["task"] != "N/A"
                        and j["state"] == "RUNNING"), None)
            envs[job["array"]] = (read_env(sib["uid"]), sib["key"]) if sib else (None, None)
        env, sib_key = envs[job["array"]]
        src = f"running task {sib_key}" if env else None
    if env is None and "created by sbatch --wrap" in script \
            and re.search(r"\bAPOLLO_DELIVERED_PACKAGE=\S", script):
        env, src = own_env(), "watcher (self-contained submit_mc.sh job)"
    if env is None:
        return None, "submit-time environment unknown (no running task of its array to read it from)"
    rewires = {}
    for j in jobs:
        if j["state"] != "PENDING" or j["dep"] in NULL or j["key"] == job["key"]:
            continue
        kind, new_dep = rewrite(j["dep"], job, jobs)
        if kind == "bad":
            return None, f"cannot rewire {j['key']}: {new_dep}"
        if kind == "doomed":
            log(f"ATTENTION {j['key']} left as is ({new_dep}); it depends on {job['key']}")
        elif kind == "ok":
            rewires[j["key"]] = (j["dep"], new_dep)
    path = os.path.join(wdir, "scripts", f"{job['key']}.sh")
    with open(path, "w") as f:
        f.write(script + "\n")
    args = ["sbatch", "--parsable", f"--job-name={d['JobName']}", "--nodes=1",
            f"--ntasks={d.get('NumTasks', '1')}", f"--partition={d['Partition']}",
            f"--chdir={d['WorkDir']}", f"--output={d['StdOut']}", "--export=ALL"]
    if d.get("CPUs/Task") not in NULL:
        args.append(f"--cpus-per-task={d['CPUs/Task']}")
    if d.get("TimeLimit") not in ("UNLIMITED", "Partition_Limit", None):
        args.append(f"--time={d['TimeLimit']}")
    if d.get("StdErr") not in NULL and d.get("StdErr") != d["StdOut"]:
        args.append(f"--error={d['StdErr']}")
    if d.get("StdIn") not in NULL and d.get("StdIn") != "/dev/null":
        args.append(f"--input={d['StdIn']}")
    if d.get("Account") not in NULL:
        args.append(f"--account={d['Account']}")
    if d.get("MinMemoryNode", "0") not in ("0", "0M"):
        args.append(f"--mem={d['MinMemoryNode']}")
    if d.get("MinMemoryCPU") not in NULL + ("0", "0M", None):
        args.append(f"--mem-per-cpu={d['MinMemoryCPU']}")
    if job["task"] != "N/A":
        args.append(f"--array={job['task']}")
    return dict(d=d, args=args, script=path, env=env, src=src, rewires=rewires), None


def restore(dep):
    return ",".join(re.sub(r"\(\w+\)$", "", t) for t in dep.split(",")
                    if not t.endswith("(fulfilled)"))


def move(job, node, p, wdir, dry):
    what = f"{job['key']} ({p['d']['JobName']})"
    args = p["args"][:1] + [f"--nodelist={node}"] + p["args"][1:] + [p["script"]]
    if dry:
        log(f"DRY-RUN would pin {what} to {node}, env from {p['src']}: {' '.join(args[1:])}")
        for k, (_, nd) in p["rewires"].items():
            log(f"DRY-RUN   {k} dependency -> {nd.replace(NEW, '<new>')}")
        return False
    rc, o, e = sh(["scontrol", "hold", job["key"]])
    if rc:
        log(f"SKIP {what}: hold failed ({e or o})")
        return False
    rc, o, e = sh(args, env=p["env"])
    if rc:
        sh(["scontrol", "release", job["key"]])
        log(f"FAILED to pin {what}: sbatch: {e or o}; released {job['key']}")
        return False
    new = o.split(";")[0]
    done = []
    for k, (old_dep, nd) in p["rewires"].items():
        nd = nd.replace(NEW, new)
        rc, o, e = sh(["scontrol", "update", f"JobId={k}", f"Dependency={nd}"])
        if rc:
            sh(["scancel", new])
            for k2, od in done:
                sh(["scontrol", "update", f"JobId={k2}", f"Dependency={restore(od)}"])
            sh(["scontrol", "release", job["key"]])
            log(f"FAILED to pin {what}: rewiring {k} ({e or o}); cancelled {new}, "
                f"restored {len(done)} dependent(s), released {job['key']}")
            return False
        done.append((k, old_dep))
        p["rewires"][k] = (old_dep, nd)
    sh(["scancel", job["key"]])
    rec = dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), old=job["key"], old_uid=job["uid"], new=new,
               node=node, name=p["d"]["JobName"], workdir=p["d"]["WorkDir"],
               stdout=p["d"]["StdOut"], env_from=p["src"],
               rewired={k: nd for k, (_, nd) in p["rewires"].items()})
    with open(os.path.join(wdir, "moves.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"PINNED {what} -> {new} on {node} (env from {p['src']})" +
        "".join(f"; {k} now {nd}" for k, nd in rec["rewired"].items()))
    return True


def watch(wdir, interval, recheck, prefix, names, once, dry):
    wdir = os.path.abspath(wdir)
    os.makedirs(os.path.join(wdir, "scripts"), exist_ok=True)
    log(f"watching {os.environ['USER']}'s jobs named {'*,'.join(names)}* for idle {prefix}* nodes: "
        f"every {interval:.0f} s, recheck {recheck:.0f} s{' (dry run)' if dry else ''}")
    told, failed, last = set(), {}, None
    while True:
        jobs, idle = snapshot(prefix)
        if jobs is None:
            log("squeue/sinfo failed; next cycle")
        else:
            wait = waiting(jobs, names)
            status = f"idle {len(idle)} {','.join(idle) or '-'} | waiting {len(wait)}"
            if status != last or (idle and wait):
                log(status)
                last = status
            if wait and (idle or dry):
                if recheck and idle:
                    time.sleep(recheck)
                    jobs, idle2 = snapshot(prefix)
                    if jobs is None:
                        idle2 = []
                    keys = {j["key"] for j in wait}
                    wait = [j for j in waiting(jobs or [], names) if j["key"] in keys]
                    idle = [n for n in idle2 if n in idle]
                nodes = list(idle) or (["<no idle node>"] if dry else [])
                envs, moved = {}, 0
                for job in wait:
                    if not nodes:
                        break
                    if time.time() - failed.get(job["key"], 0) < 1800:
                        continue        # a failed move is retried after 30 min
                    p, why = plan(job, jobs, wdir, envs)
                    if p is None:
                        if job["key"] not in told:
                            log(f"LEAVE {job['key']}: {why}")
                            told.add(job["key"])
                        continue
                    if move(job, nodes[0], p, wdir, dry):
                        nodes.pop(0)
                        moved += 1
                        jobs, _ = snapshot(prefix)
                        if jobs is None:
                            break
                        envs = {}
                    elif dry:
                        nodes.pop(0)
                    else:
                        failed[job["key"]] = time.time()
                if idle and not moved and not dry:
                    log("nothing moved this cycle")
        if once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "run":
        def opt(k, d, f=float):
            return f(a[a.index(k) + 1]) if k in a else d
        watch(a[1], opt("--interval", 120), opt("--recheck", 60), opt("--prefix", "compute", str),
              tuple(opt("--names", "a11-", str).split(",")), "--once" in a, "--dry-run" in a)
    else:
        sys.exit(__doc__)
