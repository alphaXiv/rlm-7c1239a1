# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.14",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Exact-support harnesses compose far beyond training length

    **Reproduction verdict: partially reproduced**

    This notebook is a self-contained view of the successful-run measurements
    behind the reproduction of
    [*Language model harnesses are compositional generalizers*](https://arxiv.org/abs/2607.language-model-harnesses).

    The main result is mechanistic: a harness can reuse a learned short-context
    solver at extreme depth when every chunk stays inside the complete training
    distribution and the reducer is correct. It is not a universal guarantee
    that decomposition improves every task.
    """)
    return


@app.cell
def _():
    exact_support_rows = [
        {
            "seed_set": "1729–1736",
            "indexing": "global",
            "depth": 256,
            "records": 1024,
            "exact": 0.96484375,
            "mae": 0.05078125,
            "run": "17d1e0fe",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "global",
            "depth": 256,
            "records": 1024,
            "exact": 0.6953125,
            "mae": 0.89453125,
            "run": "1e0514bf",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 256,
            "records": 1024,
            "exact": 1.0,
            "mae": 0.0,
            "run": "0ec65974",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 256,
            "records": 1024,
            "exact": 1.0,
            "mae": 0.0,
            "run": "35989087",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 1024,
            "records": 4096,
            "exact": 1.0,
            "mae": 0.0,
            "run": "a869155f",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 1024,
            "records": 4096,
            "exact": 1.0,
            "mae": 0.0,
            "run": "109065f0",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 4096,
            "records": 16384,
            "exact": 1.0,
            "mae": 0.0,
            "run": "87d9608a",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 4096,
            "records": 16384,
            "exact": 1.0,
            "mae": 0.0,
            "run": "f2a79922",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 8192,
            "records": 32768,
            "exact": 1.0,
            "mae": 0.0,
            "run": "ab064442",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 8192,
            "records": 32768,
            "exact": 1.0,
            "mae": 0.0,
            "run": "445595f1",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 16384,
            "records": 65536,
            "exact": 1.0,
            "mae": 0.0,
            "run": "7555f5cf",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 16384,
            "records": 65536,
            "exact": 1.0,
            "mae": 0.0,
            "run": "6496d012",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4",
            "depth": 32768,
            "records": 131072,
            "exact": 1.0,
            "mae": 0.0,
            "run": "daff80a6",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4",
            "depth": 32768,
            "records": 131072,
            "exact": 1.0,
            "mae": 0.0,
            "run": "f610b2e0",
        },
        {
            "seed_set": "1729–1736",
            "indexing": "local 1–4 (8 eval)",
            "depth": 65536,
            "records": 262144,
            "exact": 1.0,
            "mae": 0.0,
            "run": "96892552",
        },
        {
            "seed_set": "9001–9008",
            "indexing": "local 1–4 (8 eval)",
            "depth": 65536,
            "records": 262144,
            "exact": 1.0,
            "mae": 0.0,
            "run": "536929a9",
        },
    ]
    numeric_sum_rows = [
        {
            "depth": 8,
            "records": 128,
            "map_exact": 0.46875,
            "map_mae": 1.16015625,
            "direct_exact": 0.0,
            "direct_mae": 154.70703125,
        },
        {
            "depth": 16,
            "records": 256,
            "map_exact": 0.33984375,
            "map_mae": 2.15625,
            "direct_exact": 0.00390625,
            "direct_mae": 278.2265625,
        },
        {
            "depth": 32,
            "records": 512,
            "map_exact": 0.18359375,
            "map_mae": 4.359375,
            "direct_exact": 0.0,
            "direct_mae": 622.20703125,
        },
        {
            "depth": 64,
            "records": 1024,
            "map_exact": 0.12890625,
            "map_mae": 8.5625,
            "direct_exact": 0.0,
            "direct_mae": 1405.23046875,
        },
        {
            "depth": 128,
            "records": 2048,
            "map_exact": 0.0625,
            "map_mae": 16.98046875,
            "direct_exact": 0.0,
            "direct_mae": 2895.703125,
        },
        {
            "depth": 256,
            "records": 4096,
            "map_exact": 0.01953125,
            "map_mae": 34.40234375,
            "direct_exact": 0.0,
            "direct_mae": 6019.29296875,
        },
    ]
    return exact_support_rows, numeric_sum_rows


@app.cell
def _(mo):
    section = mo.ui.radio(
        options=[
            "Exact-support scaling",
            "Approximate numeric sum",
            "Causal checks and verdict",
        ],
        value="Exact-support scaling",
        label="Choose a result",
        inline=True,
    )
    section  # noqa: B018 - the final expression is the displayed marimo output
    return (section,)


@app.cell
def _(exact_support_rows, np, plt):
    repaired = [row for row in exact_support_rows if row["indexing"].startswith("local")]
    original = [row for row in repaired if row["seed_set"] == "1729–1736"]
    shifted = [row for row in repaired if row["seed_set"] == "9001–9008"]
    original_depths = np.array([row["depth"] for row in original])
    shifted_depths = np.array([row["depth"] for row in shifted])

    exact_figure, exact_axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    exact_axes[0].bar(
        [-0.18, 0.82], [0.96484375, 0.6953125], 0.36, color="#666666", label="Global indices"
    )
    exact_axes[0].bar([0.18, 1.18], [1.0, 1.0], 0.36, color="#009E73", label="Local 1–4")
    exact_axes[0].set_xticks([0, 1], ["Original seeds", "Shifted seeds"])
    exact_axes[0].set_ylim(0.64, 1.015)
    exact_axes[0].set_ylabel("Long exact accuracy")
    exact_axes[0].set_title("256× hidden index shift")
    exact_axes[0].legend(loc="lower left")

    exact_axes[1].plot(
        original_depths * 0.98, np.ones_like(original_depths), "o-", label="Original seeds"
    )
    exact_axes[1].plot(
        shifted_depths * 1.02, np.ones_like(shifted_depths), "s--", label="Shifted seeds"
    )
    exact_axes[1].set_xscale("log", base=2)
    exact_axes[1].set_ylim(0.985, 1.004)
    exact_axes[1].set_xlabel("Composition depth")
    exact_axes[1].set_ylabel("Long exact accuracy")
    exact_axes[1].set_title("After local renumbering")
    exact_axes[1].legend(loc="lower left")
    return (exact_figure,)


@app.cell
def _(np, numeric_sum_rows, plt):
    numeric_depths = np.array([row["depth"] for row in numeric_sum_rows])
    numeric_figure, numeric_axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    numeric_axes[0].plot(
        numeric_depths, [row["map_exact"] for row in numeric_sum_rows], "o-", label="MapReduce"
    )
    numeric_axes[0].plot(
        numeric_depths, [row["direct_exact"] for row in numeric_sum_rows], "s--", label="Direct"
    )
    numeric_axes[0].set_ylabel("Long exact accuracy")
    numeric_axes[0].set_title("Exact accuracy")
    numeric_axes[0].legend()

    numeric_axes[1].plot(
        numeric_depths, [row["map_mae"] for row in numeric_sum_rows], "o-", label="MapReduce"
    )
    numeric_axes[1].plot(
        numeric_depths, [row["direct_mae"] for row in numeric_sum_rows], "s--", label="Direct"
    )
    numeric_axes[1].set_yscale("log")
    numeric_axes[1].set_ylabel("Mean absolute error")
    numeric_axes[1].set_title("Numeric error (log scale)")
    numeric_axes[1].legend()
    for numeric_axis in numeric_axes:
        numeric_axis.set_xscale("log", base=2)
        numeric_axis.set_xticks(numeric_depths, [f"{depth}×" for depth in numeric_depths])
        numeric_axis.set_xlabel("Length multiplier")
        numeric_axis.grid(alpha=0.2)
    return (numeric_figure,)


@app.cell
def _(
    exact_figure,
    exact_support_rows,
    mo,
    numeric_figure,
    numeric_sum_rows,
    section,
):
    if section.value == "Exact-support scaling":
        content = mo.vstack(
            [
                mo.md(
                    """
                    ## Exact-support repair

                    Global record indices were an unintended distribution shift.
                    Renumbering every four-record chunk to the training-time index
                    support changes the shifted-seed 256× result from 0.6953 to 1.000
                    and preserves exact composition through 65,536×. The last endpoint
                    uses eight long examples per seed; shallower boundary runs use 32.
                    """
                ),
                exact_figure,
                mo.ui.table(exact_support_rows, pagination=True, page_size=8),
            ]
        )
    elif section.value == "Approximate numeric sum":
        content = mo.vstack(
            [
                mo.md(
                    """
                    ## Approximate local solvers

                    On Qwen2.5-7B numeric sum, local errors compound with depth, so
                    exact accuracy falls. The harness still keeps numeric error orders
                    of magnitude below direct inference at the largest tested length.
                    """
                ),
                numeric_figure,
                mo.ui.table(numeric_sum_rows, pagination=False),
            ]
        )
    else:
        content = mo.md(
            """
            ## Causal checks and verdict

            - Unseen 16-record chunks collapsed to 0 exact accuracy; four-record
              chunks reached 0.773 with the same strict length-8 solver.
            - Dropping one chunk reduced exact accuracy to 0.109; using a maximum
              reducer instead of addition produced 0.
            - On a maximum task, direct inference already generalized well, so the
              harness did not create the same advantage.

            **Verdict: partially reproduced.** The compositional mechanism is strongly
            reproduced on controlled aggregation tasks and across seed sets, but it
            depends on exact local support, solver reliability, and reducer correctness.

            **Evidence boundary:** 109 successful Kubernetes runs with terminal logs;
            NVIDIA RTX PRO 6000 Blackwell; 16 GPUs maximum concurrency; 11.885146
            observed campaign wall hours. Cancelled and marker-free runs are excluded.
            """
        )
    content  # noqa: B018 - the final expression is the displayed marimo output
    return


if __name__ == "__main__":
    app.run()
