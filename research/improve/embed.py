"""Đặc trưng mô hình nền cho từng vùng món (crop vuông quanh mask, có đệm).
Lưu 2 vector / vùng: cls (token CLS / pooled của cả crop) và mp (trung bình token patch NẰM TRONG mask).

    python embed.py dinov2b          # tham khảo: polygon nhãn tay train+val;  truy vấn: mask dự đoán (val: tr*, test: full*)
"""
import sys, os, numpy as np, cv2, torch
from transformers import AutoModel, AutoImageProcessor
from common import *

BB = {
    "dinov2b": ("facebook/dinov2-base", 14, 336),
    "dinov2l": ("facebook/dinov2-large", 14, 336),
    "dinov2s": ("facebook/dinov2-small", 14, 336),
    "dinov2s224": ("facebook/dinov2-small", 14, 224),
    "dinov2b224": ("facebook/dinov2-base", 14, 224),
    "dinov3b": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 16, 384),
    "dinov3l": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16, 384),
    "siglip2b": ("google/siglip2-base-patch16-384", 16, 384),
}
key = sys.argv[1]; hf, P, S = BB[key]
dev = "cuda"
proc = AutoImageProcessor.from_pretrained(hf)
mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
model = AutoModel.from_pretrained(hf, torch_dtype=torch.float16)
if hasattr(model, "vision_model"): model = model.vision_model
model = model.to(dev).eval()
G = S // P


def crops(img, masks, pad=0.15):
    xs, ms = [], []
    H, W = img.shape[:2]
    for m in masks:
        ys, xx = np.where(m)
        if len(ys) == 0:
            xs.append(np.zeros((S, S, 3), np.float32)); ms.append(np.full((G, G), 1 / (G * G), np.float32)); continue
        x0, x1, y0, y1 = xx.min(), xx.max() + 1, ys.min(), ys.max() + 1
        side = max(x1 - x0, y1 - y0) * (1 + 2 * pad); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(round(cx - side / 2)), int(round(cy - side / 2)); side = int(round(side))
        M = np.float32([[1, 0, -a], [0, 1, -b]])   # crop vuông, phần ngoài ảnh tô xám
        c = cv2.warpAffine(img, M, (side, side), borderValue=(124, 116, 104))
        mk = cv2.warpAffine(m.astype(np.uint8), M, (side, side))
        c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA if side > S else cv2.INTER_CUBIC)
        mk = cv2.resize(mk.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA)
        if mk.sum() < 1e-6: mk[:] = 1
        xs.append((c[:, :, ::-1].astype(np.float32) / 255 - mean) / std); ms.append(mk / mk.sum())
    return np.stack(xs).transpose(0, 3, 1, 2), np.stack(ms)


@torch.no_grad()
def embed(img, masks, bs=16):
    if len(masks) == 0: return np.zeros((0, 1)), np.zeros((0, 1))
    x, mk = crops(img, masks)
    C, Mp = [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(dev).half()
        o = model(pixel_values=xb)
        t = o.last_hidden_state.float()
        pt = t[:, -G * G:]                                   # token patch (bỏ CLS / register ở đầu)
        cls = o.pooler_output.float() if "siglip" in key else t[:, 0]
        w = torch.from_numpy(mk[i:i + bs]).to(dev).reshape(len(xb), -1, 1)
        C.append(cls.cpu().numpy()); Mp.append((pt * w).sum(1).cpu().numpy())
    return np.concatenate(C), np.concatenate(Mp)


out = f"{CACHE}/emb_{key}"; os.makedirs(out, exist_ok=True)
extra = [x.split(":") for x in sys.argv[2:]]     # vd: trE:val fullE:test -> chỉ tính các bộ mask này
# 1) tham khảo: polygon nhãn tay của train, val (và test để đo "trần" khi mask hoàn hảo)
for split in ("train", "val", "test"):
    if os.path.exists(f"{out}/gt_{split}.npz"): continue
    C, Mp, Y, I = [], [], [], []
    for im in images(split):
        a = read(split, im); g = gt(split, im, *a.shape[:2])
        c, mp = embed(a, [m for _, m in g])
        C.append(c); Mp.append(mp); Y += [k for k, _ in g]; I += [im] * len(g)
    np.savez(f"{out}/gt_{split}.npz", cls=np.concatenate(C), mp=np.concatenate(Mp), y=np.array(Y), im=np.array(I))
    print("gt", split, len(Y), flush=True)
# 2) truy vấn: mask dự đoán
for tag, split in extra or [(t, "val") for t in ("tr0", "tr1", "tr2")] + [(t, "test") for t in ("full0", "full1", "full2")]:
    if os.path.exists(f"{out}/pred_{tag}.npz"): continue
    C, Mp, I, J = [], [], [], []
    for im in images(split):
        m, _, _ = load_pred(tag, split, im)
        c, mp = embed(read(split, im), m)
        if len(m): C.append(c); Mp.append(mp)
        I += [im] * len(m); J += list(range(len(m)))
    np.savez(f"{out}/pred_{tag}.npz", cls=np.concatenate(C), mp=np.concatenate(Mp), im=np.array(I), j=np.array(J))
    print("pred", tag, len(I), flush=True)
