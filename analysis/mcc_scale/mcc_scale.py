"""Midcourse-correction burns against dispersion scale (JBIS §4.2 TODO, 2026-10-31).

    venv/bin/python analysis/mcc_scale/mcc_scale.py LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Translunar: trials that completed TLI (tlmcc_n_exec not null); burns tlmcc_n_exec, total dv mcc_dv_ms.
Trans-Earth: trials that executed TEI (temcc_n_exec not null); burns temcc_n_exec, total dv mcc_total_dv_ms.
"""
import sys

import numpy as np
import pandas as pd

rows = []
for arg in sys.argv[1:]:
    lab, run = arg.split("=", 1)
    d = pd.read_csv(f"{run}/results.csv", low_memory=False)
    tl = d[d.tlmcc_n_exec.notna()]
    te = d[d.temcc_n_exec.notna()]
    for leg, x, n, dv in (("translunar", tl, "tlmcc_n_exec", "mcc_dv_ms"),
                          ("trans-Earth", te, "temcc_n_exec", "mcc_total_dv_ms")):
        b = x[n].astype(int)
        v = x[dv].astype(float)
        rows.append(dict(scale=lab, leg=leg, trials=len(x), mean_burns=b.mean(),
                         p0=100 * (b == 0).mean(), p1=100 * (b == 1).mean(), p2=100 * (b == 2).mean(),
                         p3plus=100 * (b >= 3).mean(), dv_median=v.median(), dv_p95=v.quantile(.95),
                         dv_p99=v.quantile(.99), dv_max=v.max(), over30=int((v > 30).sum())))
t = pd.DataFrame(rows).sort_values(["leg", "scale"], ascending=[False, True])
pd.set_option("display.width", 200)
print(t.to_string(index=False, float_format=lambda f: f"{f:.2f}"))
