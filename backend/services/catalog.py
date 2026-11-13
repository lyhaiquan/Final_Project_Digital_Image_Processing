"""Danh mục 45 món + công thức gram/kcal. Chỉ đọc file trong models/, không cần nạp mô hình (FR-B11, FR-B18)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from calo_khay.pipeline import estimate_grams


class Catalog:
    def __init__(self, models_dir: Path):
        self.gm = json.loads((models_dir / "gram_model.json").read_text(encoding="utf-8"))
        self.rows: dict[str, dict] = {}
        with open(models_dir / "calories.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kcal = r.get("kcal_per_100g") or ""
                self.rows[r["class"]] = dict(
                    id=int(r["id"]), cls=r["class"], name=r["name_vi"],
                    kcal_per_100g=float(kcal) if kcal and not math.isnan(float(kcal)) else 0.0,  # đồ uống: 0 (QT-05)
                    source=r.get("source") or None)

    def __contains__(self, cls: str) -> bool:
        return cls in self.rows

    def name(self, cls: str) -> str:
        return self.rows[cls]["name"] if cls in self.rows else cls

    def default_g(self, cls: str) -> int:
        """Khẩu phần trung bình (QT-10); cơm 228 g (QT-04)."""
        if cls == "rice":
            return round(self.gm["rice_mean_g"])
        return round(self.gm["class_mean_g"].get(cls, self.gm["global_mean_g"]))

    def grams(self, cls: str, area_ratio: float | None, rice_volume: float | None = None) -> tuple[int, str]:
        g, how = estimate_grams(self.gm, cls, area_ratio, rice_volume if cls == "rice" else None)
        return round(g), how

    def kcal(self, cls: str, grams: float) -> int:
        return round(grams * self.rows[cls]["kcal_per_100g"] / 100)

    def suggestions(self, cls: str) -> list[str]:
        return list(self.gm.get("confusions", {}).get(cls, []))

    def classes(self) -> list[dict]:
        return [dict(cls=r["cls"], name=r["name"], kcal_per_100g=r["kcal_per_100g"], default_g=self.default_g(r["cls"]),
                     source=r["source"], suggestions=self.suggestions(r["cls"]))
                for r in sorted(self.rows.values(), key=lambda r: r["id"])]
