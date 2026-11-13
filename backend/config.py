"""Cấu hình backend, đọc từ biến môi trường hoặc file .env (xem .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", env_prefix="CALO_", extra="ignore",
                                      env_ignore_empty=True)  # "CALO_DEVICE=" để trống = dùng mặc định

    # --- cơ sở dữ liệu ---
    database_url: str = "mysql+pymysql://calo:calo@localhost:3306/calo_khay?charset=utf8mb4"

    # --- mô hình ---
    models_dir: Path = ROOT / "models"
    engine: str = Field("real", description="real = chạy CaloKhay; mock = trả mock/scan.json (dev frontend, test)")
    device: str | None = Field(None, description="cpu, 0 (GPU số 0)... Bỏ trống = tự chọn")
    use_dino: bool = True
    depth_enabled: bool = Field(False, description="nạp Depth-Anything-V2 và bật độ sâu cho cơm mặc định")
    flag_conf: float | None = Field(None, description="ngưỡng 'Cần xác nhận'; bỏ trống = 0.75 khi có DINO, 0.5 khi không")
    model_version: str | None = Field(None, description="bỏ trống = tự đặt theo tên model")
    warmup_image: Path = ROOT / "examples" / "B_170.jpg"
    mock_scan: Path = ROOT / "backend" / "mock" / "scan.json"

    # --- ảnh ---
    max_image_mb: float = 10
    min_image_side: int = 320
    blur_threshold: float = Field(40.0, description="phương sai Laplacian dưới ngưỡng -> cảnh báo blurry")

    # --- hàng chờ, lưu tạm ---
    max_queue: int = Field(8, description="số ảnh được chờ tối đa (không tính ảnh đang chạy)")
    scan_ttl_hours: float = 24
    cleanup_interval_s: int = 3600
    data_dir: Path = ROOT / "data"

    # --- web ---
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    frontend_dist: Path | None = Field(None, description="thư mục frontend đã build; có thì phục vụ luôn tại /")
    timezone: str = "Asia/Ho_Chi_Minh"
    daily_kcal: int = 2000
    error_pct: int = 30

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def corrections_dir(self) -> Path:
        return self.data_dir / "corrections"


@lru_cache
def get_settings() -> Settings:
    return Settings()
