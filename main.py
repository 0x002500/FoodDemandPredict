"""Append seven production demand forecasts to every sheet in an XLSX workbook."""
from __future__ import annotations

import argparse
from pathlib import Path

from data.xlsx_io import append_forecasts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("饭堂销量样例_2026-08-01至09-30.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("饭堂销量预测_未来7天.xlsx"))
    parser.add_argument("--days", type=int, default=7, help="要追加的未来日期数")
    parser.add_argument("--device", default="cpu", help="torch 设备，例如 cpu 或 cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = append_forecasts(args.input, args.output, days=args.days, device=args.device)
    print(f"已生成 {report.output_path}，共追加 {report.forecast_rows} 条预测行。")


if __name__ == "__main__":
    main()
