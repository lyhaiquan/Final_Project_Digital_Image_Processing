"""Chấm nhận diện trên TEST (một lần) với cấu hình đã chọn trên val."""
import sys, numpy as np, pandas as pd
from common import *
import eval_cls as E, postproc as PP
E.key = "dinov2b224"
CFG = dict(w=0.8, t_low=0.1, nms=0.7, gate=0.5)
rows = []
for split, tags, ens, ref in (("val", ("tr0", "tr1", "tr2"), "trE", ("train",)), ("test", ("full0", "full1", "full2"), "fullE", ("train", "val"))):
    _, Ps = E.run(split, tags + (ens,), ref, ("mp", "lr", 100.0))
    b = [score(split, t) for t in tags]
    r1 = [score(split, t, cls_of=E.relabel(t, split, Ps[t], 0.8)) for t in tags]
    m, per = PP.evaluate(split, tags, Ps, **CFG)
    keys = ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")
    avg = lambda L: {k: round(float(np.mean([x[k] for x in L])), 1) for k in keys}
    rows += [dict(tap=split, cach="YOLO (TB 3 seed)", **avg(b)), dict(tap=split, cach="+ DINO đổi nhãn (TB 3 seed)", **avg(r1)),
             dict(tap=split, cach="+ DINO + hạ ngưỡng (TB 3 seed)", **m)]
    for t, x, y in zip(tags, b, per): rows.append(dict(tap=split, cach=f"   {t}: YOLO -> DINO+hạ ngưỡng", dung=f"{x['dung']}->{y['dung']}", nham=f"{x['nham']}->{y['nham']}", sot=f"{x['sot']}->{y['sot']}", F1=f"{x['F1']}->{y['F1']}", fp=f"{x['fp']}->{y['fp']}"))
    rows.append(dict(tap=split, cach="gộp 3 seed, nhãn YOLO", **{k: score(split, ens)[k] for k in keys}))
    rows.append(dict(tap=split, cach="gộp 3 seed + DINO + hạ ngưỡng", **PP.evaluate(split, (ens,), Ps, **CFG)[0]))
pd.set_option("display.width", 250); d = pd.DataFrame(rows); print(d.to_string(index=False)); d.to_csv(f"{CACHE}/../result_recognition.csv", index=False)
