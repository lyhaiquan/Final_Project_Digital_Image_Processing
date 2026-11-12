"""Sinh models/benchmark.json cho GET /api/benchmark (FR-F16) từ các CSV kết quả; không gõ tay số nào.

    python tools/build_benchmark_json.py
Chạy lại mỗi khi đổi model hoặc cập nhật docs/results/, research/improve/.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = os.path.join(ROOT, "docs", "results")
IMP = os.path.join(ROOT, "research", "improve")


def r(x, d=1):
    return None if pd.isna(x) else round(float(x), d)


def main():
    cal = pd.read_csv(os.path.join(ROOT, "models", "calories.csv")).set_index("class")
    name = cal.name_vi.to_dict()

    # --- phân đoạn: trung bình 3 seed của food_seg_v4_final
    seg = pd.read_csv(os.path.join(R, "seg_v4_final_summary.csv"))
    ap = pd.read_csv(os.path.join(R, "seg_v4_final_per_class_ap_s0_s2.csv")).merge(
        pd.read_csv(os.path.join(R, "seg_v4_final_per_class_ap_s1.csv")), on="lop")
    ap["ap"] = ap[["full_s0", "full_s1", "full_s2"]].mean(axis=1)
    per_class = [dict(cls=c, name=name.get(c, c), ap=r(v, 3)) for c, v in ap.sort_values("ap", ascending=False)[["lop", "ap"]].values]

    # --- gram: khẩu phần TB vs độ sâu cho cơm (pipeline thật, mask dự đoán)
    dep = pd.read_csv(os.path.join(R, "depth_eval_pred.csv")).set_index("com")
    no_d, d = dep.loc["khẩu phần TB"], dep.loc["độ sâu"]

    # --- đóng góp từng cải tiến (YOLO -> +DINOv2), có và không có độ sâu
    pipe = pd.read_csv(os.path.join(IMP, "result_pipeline.csv"), index_col=0)
    pipe_d = pd.read_csv(os.path.join(IMP, "result_pipeline_depth.csv"), index_col=0)

    # --- các cách tính gram so với mốc (mask dự đoán)
    gs = pd.read_csv(os.path.join(R, "summary.csv"))
    gs = gs[gs["mask"] == "PRED"]
    label = {"mean": "khẩu phần trung bình (mốc)", "ratio": "tỉ lệ diện tích", "loglog": "log-log diện tích/đĩa",
             "hybrid": "kết hợp"}

    # --- cặp món hay nhầm trên tập test
    cf = pd.read_csv(os.path.join(R, "confusion_test.csv"))
    wrong = cf[(cf.that != cf.doan) & (cf.doan != "(bỏ sót)")]
    pairs = wrong.groupby(["that", "doan"]).size().sort_values(ascending=False).head(10)

    out = dict(
        model="food_seg_v4_final + DINOv2",
        headline=dict(
            mask_map50_95=r(seg.test_mask_mAP50_95.mean(), 3),
            item_mae_g=r(pipe.loc["MAE món (g)", "mới (+DINOv2)"]),
            tray_mae_g=r(pipe.loc["MAE gram khay", "mới (+DINOv2)"]),
            tray_kcal_medape_pct=r(pipe.loc["medAPE kcal khay %", "mới (+DINOv2)"]),
        ),
        depth=dict(
            off=dict(item_mae_g=r(no_d.test_MAE), rice_mae_g=r(no_d.test_MAE_com), tray_mae_g=r(no_d.test_khay_MAE)),
            on=dict(item_mae_g=r(d.test_MAE), rice_mae_g=r(d.test_MAE_com), tray_mae_g=r(d.test_khay_MAE)),
        ),
        improvements=[dict(metric=k, yolo=r(v["cũ (chỉ YOLO)"]), dino=r(v["mới (+DINOv2)"]),
                           yolo_depth=r(pipe_d.loc[k, "cũ (chỉ YOLO)"]) if k in pipe_d.index else None,
                           dino_depth=r(pipe_d.loc[k, "mới (+DINOv2)"]) if k in pipe_d.index else None)
                      for k, v in pipe.iterrows()],
        gram_methods=[dict(method=m, label=label.get(m, m), item_mae_g=r(a), tray_mae_g=r(b))
                      for m, a, b in gs[["method", "test_MAE_g", "test_tray_MAE_g"]].values],
        per_class_ap=per_class,
        confusions=[dict(true=t, true_name=name.get(t, t), pred=p, pred_name=name.get(p, p), count=int(n))
                    for (t, p), n in pairs.items()],
    )
    dst = os.path.join(ROOT, "models", "benchmark.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("Đã ghi", dst)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
