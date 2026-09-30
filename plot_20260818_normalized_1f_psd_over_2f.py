#!/usr/bin/env python3
"""Plot normalized 1f noise spectra: ASD(1f) / measured 2f.

The project convention calls the ASD (V/sqrt(Hz)) the PSD.  Here the ASD
spectra from the three acquisitions at each operating point are averaged and
then divided by the corresponding 2f voltage.  Temperature is encoded by
color; the six powers are encoded by line style.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

from analyze_20260818_1f2f_asd import (
    CALIBRATION,
    DATASETS,
    FS_HZ,
    acquisition_path,
    asd,
)

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "analysis_output_20260818_1f2f_asd"
POWER_POINTS_MW = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
COLORS = {"28.6 °C": "#236b8e", "180 °C": "#c43c39"}
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
CHINESE_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def configure_style() -> None:
    # Matplotlib's default font lacks CJK glyphs; register the system CJK font
    # explicitly so labels remain readable in the raster output.
    font_manager.fontManager.addfont(CHINESE_FONT)
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )


def normalized_spectra() -> dict[str, dict[float, dict[str, np.ndarray]]]:
    result: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    for name, directory in DATASETS.items():
        result[name] = {}
        for power in POWER_POINTS_MW:
            two_f_mv = CALIBRATION[name][power][1]
            two_f_v = two_f_mv / 1000.0
            spectra = []
            frequency_axis = None
            for run in range(1, 4):
                frequency, spectrum = asd(
                    np.loadtxt(acquisition_path(directory, power, run), dtype=float)
                )
                if frequency_axis is None:
                    frequency_axis = frequency
                spectra.append(np.interp(frequency_axis, frequency, spectrum))
            assert frequency_axis is not None
            mean_asd = np.mean(np.asarray(spectra), axis=0)
            std_asd = np.std(np.asarray(spectra), axis=0, ddof=1)
            result[name][power] = {
                "frequency_hz": frequency_axis,
                "mean_asd": mean_asd,
                "std_asd": std_asd,
                "two_f_v": np.array(two_f_v),
                "normalized_mean": mean_asd / two_f_v,
                "normalized_std": std_asd / two_f_v,
            }
    return result


def save_csv(data: dict[str, dict[float, dict[str, np.ndarray]]]) -> Path:
    target = OUTDIR / "归一化1f_PSD除以2f_频谱.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "条件",
                "probe功率_mW",
                "2f_mV",
                "频率_Hz",
                "三组平均1f_PSD_V每根号Hz",
                "三组平均PSD除以2f_Hz负二分之一",
                "三组PSD样本标准差除以2f_Hz负二分之一",
            ]
        )
        for name, by_power in data.items():
            for power, values in by_power.items():
                for frequency, mean, std in zip(
                    values["frequency_hz"],
                    values["mean_asd"],
                    values["normalized_std"],
                ):
                    writer.writerow(
                        [
                            name,
                            f"{power:.1f}",
                            f"{float(values['two_f_v']) * 1000.0:.1f}",
                            f"{frequency:.8g}",
                            f"{mean:.10e}",
                            f"{(mean / float(values['two_f_v'])):.10e}",
                            f"{std:.10e}",
                        ]
                    )
    return target


def plot(data: dict[str, dict[float, dict[str, np.ndarray]]]) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    for name in DATASETS:
        for index, power in enumerate(POWER_POINTS_MW):
            values = data[name][power]
            ax.loglog(
                values["frequency_hz"],
                values["normalized_mean"],
                color=COLORS[name],
                linestyle=LINESTYLES[index],
                linewidth=1.25,
                alpha=0.92,
            )

    ax.set_xlim(0.15, 300.0)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("1f PSD / 2f (Hz$^{-1/2}$)")
    ax.set_title("20260818：归一化 1f PSD / 2f 频谱（相同 probe 功率）")
    ax.grid(True, which="both", alpha=0.23, linewidth=0.45)

    condition_handles = [
        Line2D([0], [0], color=COLORS[name], lw=2.0, label=name) for name in DATASETS
    ]
    power_handles = [
        Line2D([0], [0], color="#333333", lw=1.5, linestyle=LINESTYLES[i], label=f"{p:g} mW")
        for i, p in enumerate(POWER_POINTS_MW)
    ]
    first = ax.legend(handles=condition_handles, title="温度", loc="upper right", fontsize=8)
    ax.add_artist(first)
    ax.legend(handles=power_handles, title="probe 功率", loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    target = OUTDIR / "归一化1f_PSD除以2f_频谱.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def write_report(csv_path: Path, plot_path: Path) -> Path:
    target = OUTDIR / "归一化1f_PSD除以2f_说明.md"
    target.write_text(
        "\n".join(
            [
                "# 归一化 1f PSD / 2f 频谱",
                "",
                "- 采样率：837.1 Hz；三组原始记录分别计算 ASD，再取三组平均。项目沿用习惯，将 ASD（V/sqrt(Hz)）称为 PSD。",
                "- 归一化量为 `PSD(1f)/2f`，其中 2f 使用图片标定值并换算为 V。因此纵轴单位为 `Hz^-1/2`；电压单位在相除时抵消。",
                "- 两个条件共同选取六个 probe 功率工作点：0.5、1.0、1.5、2.0、2.5、3.0 mW。颜色表示温度，线型表示功率。",
                "- 28.6 ℃和 180 ℃的 2f 均使用修正后的图片转录值；1f 数值不参与本图。",
                "- 本图是在相同 probe 功率下比较 `PSD/2f`；两温度的 2f-功率对应关系不同，因此不能直接把本图的上下关系解释为相同 2f 横坐标下的 2f-ASD 曲线排序。",
                "",
                f"图：`{plot_path.name}`",
                f"数据：`{csv_path.name}`",
                "补充关系图：`2f_vs_1f_PSD除以2f_模型拟合.png`；模型参数：`2f_vs_1f_PSD除以2f_模型拟合.csv`。",
                "补充频带关系图：`2f_vs_1f_PSD除以2f_频带模型拟合.png`；模型参数：`2f_vs_1f_PSD除以2f_频带模型拟合.csv`。",
                "",
                "该图用于比较归一化噪声谱形状；它不等同于对 1f 频率分量本身做窄带锁相解调。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    configure_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = normalized_spectra()
    csv_path = save_csv(data)
    plot_path = plot(data)
    report_path = write_report(csv_path, plot_path)
    print(f"Plot written to {plot_path}")
    print(f"CSV written to {csv_path}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
