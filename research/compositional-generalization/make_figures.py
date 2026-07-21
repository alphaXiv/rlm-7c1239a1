"""Generate the static figures used by the reproduction report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def exact_support_figure() -> None:
    depths = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    exact = np.ones_like(depths, dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    labels = ["Original seeds\n1729–1736", "Shifted seeds\n9001–9008"]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, [0.96484375, 0.6953125], width, color=GRAY, label="Global indices")
    axes[0].bar(x + width / 2, [1.0, 1.0], width, color=GREEN, label="Local 1–4 indices")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.64, 1.015)
    axes[0].set_ylabel("Long exact accuracy")
    axes[0].set_title("At 256×, prompt support—not chunk length—matters")
    axes[0].legend(loc="lower left")
    for position, value in zip(x - width / 2, [0.96484375, 0.6953125], strict=True):
        axes[0].text(position, value + 0.008, f"{value:.3f}", ha="center", va="bottom")
    for position in x + width / 2:
        axes[0].text(position, 1.002, "1.000", ha="center", va="bottom")

    axes[1].plot(depths * 0.98, exact, "o-", color=BLUE, label="Original seeds")
    axes[1].plot(depths * 1.02, exact, "s--", color=ORANGE, label="Shifted seeds")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0.985, 1.004)
    axes[1].set_yticks([0.99, 0.995, 1.0])
    axes[1].set_xticks(
        [256, 1024, 4096, 16384, 65536], ["256×", "1,024×", "4,096×", "16,384×", "65,536×"]
    )
    axes[1].set_xlabel("Composition depth (four-record chunks)")
    axes[1].set_ylabel("Long exact accuracy")
    axes[1].set_title("Local renumbering sustains exact composition")
    axes[1].legend(loc="lower left")
    axes[1].annotate(
        "32 examples/seed through 32,768×\n8 examples/seed at 65,536×",
        xy=(65536, 0.9985),
        xytext=(7500, 0.989),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
    )

    figure.savefig(OUTPUT_DIR / "exact-support-scaling.png", bbox_inches="tight")
    plt.close(figure)


def numeric_sum_figure() -> None:
    depths = np.array([8, 16, 32, 64, 128, 256])
    map_exact = np.array([0.46875, 0.33984375, 0.18359375, 0.12890625, 0.0625, 0.01953125])
    direct_exact = np.array([0.0, 0.00390625, 0.0, 0.0, 0.0, 0.0])
    map_mae = np.array([1.16015625, 2.15625, 4.359375, 8.5625, 16.98046875, 34.40234375])
    direct_mae = np.array(
        [154.70703125, 278.2265625, 622.20703125, 1405.23046875, 2895.703125, 6019.29296875]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(depths, [f"{depth}×" for depth in depths])
        axis.set_xlabel("Length multiplier")

    axes[0].plot(depths, map_exact, "o-", color=BLUE, label="MapReduce harness")
    axes[0].plot(depths, direct_exact, "s--", color=ORANGE, label="Direct inference")
    axes[0].set_ylim(-0.015, 0.5)
    axes[0].set_ylabel("Long exact accuracy")
    axes[0].set_title("Residual local errors compound with depth")
    axes[0].legend()

    axes[1].plot(depths, map_mae, "o-", color=BLUE, label="MapReduce harness")
    axes[1].plot(depths, direct_mae, "s--", color=ORANGE, label="Direct inference")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Long mean absolute error (log scale)")
    axes[1].set_title("The harness keeps numeric error far lower")
    axes[1].legend()

    figure.savefig(OUTPUT_DIR / "numeric-sum-scaling.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    exact_support_figure()
    numeric_sum_figure()


if __name__ == "__main__":
    main()
