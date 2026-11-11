"""Ngưỡng conf: đổi bỏ sót lấy món 'ma' như thế nào? (val, 3 model chỉ-train, nhãn YOLO vs nhãn kết hợp DINOv2)"""
import numpy as np, pandas as pd
from common import *
import eval_cls as E
E.key = "dinov2b"
rows = []
_, Ps = E.run("val", ("tr0", "tr1", "tr2"), ("train",), ("mp", "lr", 100.0))
for conf in (0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
    for name, f in (("YOLO", lambda t: None), ("DINO w.8", lambda t: E.relabel(t, "val", Ps[t], 0.8))):
        sc = [score("val", t, cls_of=f(t), conf=conf) for t in ("tr0", "tr1", "tr2")]
        rows.append(dict(conf=conf, nhan=name, **{k: round(np.mean([b[k] for b in sc]), 1) for k in ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")}))
print(pd.DataFrame(rows).to_string(index=False))
