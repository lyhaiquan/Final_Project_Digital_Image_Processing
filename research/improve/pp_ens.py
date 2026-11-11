import sys, numpy as np, pandas as pd
from common import *
import eval_cls as E, postproc as PP
E.key = "dinov2b"
tags = ("trE",)
_, Ps = E.run("val", tags, ("train",), ("mp", "lr", 100.0))
rows = [dict(cach="gộp 3 seed, nhãn YOLO, conf 0.25", **score("val", "trE"))]
for t_low in (0.25, 0.15, 0.1, 0.05):
    for gate in (0.0, 0.5):
        rows.append(dict(cach=f"gộp 3 seed + DINO, t_low={t_low} gate={gate}", **PP.evaluate("val", tags, Ps, w=0.8, t_low=t_low, nms=0.7, gate=gate)[0]))
pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
