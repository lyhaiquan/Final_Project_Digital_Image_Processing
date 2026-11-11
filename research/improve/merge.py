"""Thử nghiệm 2: gộp dự đoán của nhiều nguồn (TTA lật / ảnh 800, hoặc 3 seed) thành một bộ mask mới.
Gom cụm tham lam theo IoU >= 0.5 (không phân biệt lớp). Mỗi cụm: điểm = tổng conf cao nhất của từng nguồn / số nguồn,
lớp = lớp có tổng conf lớn nhất, mask = bỏ phiếu pixel có trọng số conf.
    python merge.py
"""
import numpy as np
from common import *

SETS = {}
for i in range(3):
    SETS[f"tr{i}F"] = ("val", [(f"tr{i}", "n640"), (f"tr{i}", "f640")])
    SETS[f"tr{i}T"] = ("val", [(f"tr{i}", "n640"), (f"tr{i}", "f640"), (f"tr{i}", "n800")])
    SETS[f"full{i}F"] = ("test", [(f"full{i}", "n640"), (f"full{i}", "f640")])
    SETS[f"full{i}T"] = ("test", [(f"full{i}", "n640"), (f"full{i}", "f640"), (f"full{i}", "n800")])
SETS["trE"] = ("val", [(f"tr{i}", "n640") for i in range(3)])
SETS["trEF"] = ("val", [(f"tr{i}", a) for i in range(3) for a in ("n640", "f640")])
SETS["fullE"] = ("test", [(f"full{i}", "n640") for i in range(3)])
SETS["fullEF"] = ("test", [(f"full{i}", a) for i in range(3) for a in ("n640", "f640")])


def merge(split, im, sources, thr=0.5):
    M, C, S, Q = [], [], [], []
    for q, (tag, aug) in enumerate(sources):
        m, c, s = load_pred(tag, split, im, aug)
        M.append(m); C += list(c); S += list(s); Q += [q] * len(c)
    M = np.concatenate(M); C, S, Q = np.array(C), np.array(S), np.array(Q)
    H, W = M.shape[1:]
    if len(M) == 0: return M, C, S
    x = M.reshape(len(M), -1).astype(np.float32); a = x.sum(1); inter = x @ x.T
    iou = inter / np.maximum(a[:, None] + a[None] - inter, 1)
    free = np.ones(len(M), bool); om, oc, os_ = [], [], []
    for i in np.argsort(-S):
        if not free[i]: continue
        mem = np.where(free & (iou[i] >= thr))[0]; free[mem] = False
        score = sum(S[mem][Q[mem] == q].max() for q in set(Q[mem].tolist())) / len(sources)
        votes = np.zeros(K); np.add.at(votes, C[mem], S[mem])
        wm = (x[mem] * S[mem, None]).sum(0) >= 0.5 * S[mem].sum()
        if wm.sum() == 0: continue
        om.append(wm.reshape(H, W)); oc.append(votes.argmax()); os_.append(score)
    return np.array(om), np.array(oc), np.array(os_)


if __name__ == "__main__":
    for name, (split, src) in SETS.items():
        if os.path.isdir(f"{CACHE}/pred/{name}_n640"): continue
        for im in images(split):
            m, c, s = merge(split, im, src)
            k = s >= 0.05
            save_pred(name, split, im, "n640", m[k], c[k], s[k])
        print(name, score(split, name), flush=True)
