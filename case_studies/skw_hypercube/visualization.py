"""
Visualization utilities for the SKW QSMEF case study.

This module contains functions for:

    - printing Shapley-value tables,
    - plotting Shapley contribution profiles,
    - comparing the oracle contribution between
      the correct and modified implementations.
"""

import numpy as np
import matplotlib.pyplot as plt


def print_shapley_table(
    results,
    start=15,
    end=23,
    decimals=6
):
    """
    Print the Shapley-value table for a selected step window.
    """

    steps = results["steps"]

    phi_O = results["phi"]["O"]
    phi_G = results["phi"]["G"]
    phi_S = results["phi"]["S"]

    sum_phi = phi_O + phi_G + phi_S

    delta_E = results["delta_E"]

    error = sum_phi - delta_E

    mask = (
        (steps >= start)
        & (steps <= end)
    )

    print("\n" + "=" * 105)

    print(
        "SHAPLEY TABLE FOR THE FUNCTIONAL METRIC"
        f" | {results['label']}"
    )

    print("=" * 105)

    print(
        f"{'t':>4} | "
        f"{'phi_O':>12} | "
        f"{'phi_G':>12} | "
        f"{'phi_S':>12} | "
        f"{'sum_phi':>12} | "
        f"{'Delta_E':>12} | "
        f"{'error':>12}"
    )

    print("-" * 105)

    for t, o, g, s, total, de, err in zip(
        steps[mask],
        phi_O[mask],
        phi_G[mask],
        phi_S[mask],
        sum_phi[mask],
        delta_E[mask],
        error[mask]
    ):

        print(
            f"{int(t):>4} | "
            f"{o:>12.{decimals}f} | "
            f"{g:>12.{decimals}f} | "
            f"{s:>12.{decimals}f} | "
            f"{total:>12.{decimals}f} | "
            f"{de:>12.{decimals}f} | "
            f"{err:>12.2e}"
        )

    print("-" * 105)

    print(
        "Maximum efficiency error "
        f"|Σφ - ΔE| = "
        f"{np.max(np.abs(error)):.3e}"
    )

    print("=" * 105)


def signed_stacked_bars(
    ax,
    x,
    series,
    labels,
    colors,
    width=0.72
):
    """
    Draw stacked bars while treating positive and negative
    contributions independently.
    """

    positive_bottom = np.zeros(len(x))
    negative_bottom = np.zeros(len(x))

    for values, label, color in zip(
        series,
        labels,
        colors
    ):

        values = np.asarray(values)

        bottom = np.where(
            values >= 0,
            positive_bottom,
            negative_bottom
        )

        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.6
        )

        positive_bottom = np.where(
            values >= 0,
            positive_bottom + values,
            positive_bottom
        )

        negative_bottom = np.where(
            values < 0,
            negative_bottom + values,
            negative_bottom
        )


def plot_shapley_profile(
    results,
    start=15,
    end=23,
    title=None,
    filename_prefix="skw_profile",
    save=True
):
    """
    Plot the Shapley contribution profile for the SKW case study.
    """

    steps = results["steps"]

    phi_O = results["phi"]["O"]
    phi_G = results["phi"]["G"]
    phi_S = results["phi"]["S"]

    sum_phi = phi_O + phi_G + phi_S

    delta_E = results["delta_E"]

    t_opt = results["t_opt"]

    mask = (
        (steps >= start)
        & (steps <= end)
    )

    t = steps[mask]

    phi_O = phi_O[mask]
    phi_G = phi_G[mask]
    phi_S = phi_S[mask]

    sum_phi = sum_phi[mask]

    delta_E = delta_E[mask]

    max_error = np.max(
        np.abs(
            sum_phi - delta_E
        )
    )

    if title is None:

        title = (
            "QSMEF: Shapley profile"
            f" | {results['label']}"
        )

    fig, ax = plt.subplots(
        figsize=(10.5, 5.8)
    )

    signed_stacked_bars(
        ax,
        t,
        [
            phi_O,
            phi_G,
            phi_S
        ],
        [
            r"$\phi_O(t)$ Oracle",
            r"$\phi_G(t)$ Grover coin",
            r"$\phi_S(t)$ Shift"
        ],
        [
            "#D55E00",
            "#56B4E9",
            "#009E73"
        ]
    )

    ax.plot(
        t,
        delta_E,
        "-o",
        color="black",
        linewidth=2.4,
        markersize=6,
        label=r"$\Delta E_t$",
        zorder=5
    )

    ax.plot(
        t,
        sum_phi,
        "--x",
        color="gray",
        linewidth=2,
        markersize=7,
        label=r"$\sum_b \phi_b(t)$",
        zorder=6
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.9
    )

    ax.axvline(
        t_opt,
        linestyle="--",
        color="red",
        linewidth=1.8,
        alpha=0.75
    )

    fig.canvas.draw()

    y_min, y_max = ax.get_ylim()

    ax.text(
        t_opt + 0.12,
        y_max - 0.06 * (y_max - y_min),
        rf"$t_{{\mathrm{{opt}}}}={t_opt}$",
        color="red",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top"
    )

    residual_text = (
        rf"$\max_t\left|"
        rf"\sum_b \phi_b(t)-\Delta E_t"
        rf"\right|={max_error:.2e}$"
    )

    ax.text(
        0.985,
        0.965,
        residual_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            edgecolor="gray",
            boxstyle="round,pad=0.25"
        ),
        fontsize=10,
        zorder=10
    )

    ax.set_xlabel(
        r"Step $t$",
        fontsize=12
    )

    ax.set_ylabel(
        "Shapley value / functional metric variation",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=10
    )

    ax.set_xticks(t)

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=9,
        framealpha=0.95
    )

    plt.tight_layout()

    if save:

        png = f"{filename_prefix}.png"

        pdf = f"{filename_prefix}.pdf"

        fig.savefig(
            png,
            dpi=600,
            bbox_inches="tight"
        )

        fig.savefig(
            pdf,
            bbox_inches="tight"
        )

        print(
            f"Files saved: {png}, {pdf}"
        )

    plt.show()

    plt.close(fig)


def print_oracle_comparison(
    correct_results,
    modified_results,
    start=15,
    end=23
):
    """
    Print a comparison of the oracle contribution between
    the correct and modified implementations.
    """

    steps = correct_results["steps"]

    mask = (
        (steps >= start)
        & (steps <= end)
    )

    t = steps[mask]

    oracle_correct = (
        correct_results["phi"]["O"][mask]
    )

    oracle_modified = (
        modified_results["phi"]["O"][mask]
    )

    difference = (
        oracle_modified
        - oracle_correct
    )

    print("\n" + "=" * 90)

    print(
        "COMPARATIVE SUMMARY OF THE ORACLE CONTRIBUTION"
    )

    print("=" * 90)

    print(
        f"{'t':>4} | "
        f"{'correct phi_O':>16} | "
        f"{'modified phi_O':>18} | "
        f"{'difference':>14}"
    )

    print("-" * 90)

    for t_i, correct, modified, diff in zip(
        t,
        oracle_correct,
        oracle_modified,
        difference
    ):

        print(
            f"{int(t_i):>4} | "
            f"{correct:>16.6f} | "
            f"{modified:>18.6f} | "
            f"{diff:>14.6f}"
        )

    print("-" * 90)

    print(
        "Maximum absolute difference in the window = "
        f"{np.max(np.abs(difference)):.6f}"
    )

    print("=" * 90)


def plot_oracle_comparison(
    correct_results,
    modified_results,
    start=15,
    end=23,
    filename_prefix="oracle_phase_comparison",
    save=True
):
    """
    Compare the oracle contribution between the correct
    implementation and the implementation with a modified phase.
    """

    steps = correct_results["steps"]

    mask = (
        (steps >= start)
        & (steps <= end)
    )

    t = steps[mask]

    oracle_correct = (
        correct_results["phi"]["O"][mask]
    )

    oracle_modified = (
        modified_results["phi"]["O"][mask]
    )

    difference = (
        oracle_modified
        - oracle_correct
    )

    fig, ax = plt.subplots(
        figsize=(10.2, 5.2)
    )

    ax.plot(
        t,
        oracle_correct,
        "-o",
        color="#D55E00",
        linewidth=2.5,
        markersize=6,
        label=r"$\phi_O(t)$ correct implementation"
    )

    ax.plot(
        t,
        oracle_modified,
        "--s",
        color="#0072B2",
        linewidth=2.5,
        markersize=6,
        label=r"$\phi_O(t)$ modified oracle phase"
    )

    ax.fill_between(
        t,
        oracle_correct,
        oracle_modified,
        color="gray",
        alpha=0.18,
        label="Difference between profiles"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.9
    )

    ax.axvline(
        correct_results["t_opt"],
        linestyle="--",
        color="red",
        linewidth=1.8,
        alpha=0.75
    )

    if correct_results["t_opt"] in t:

        index = int(
            np.where(
                t
                == correct_results["t_opt"]
            )[0][0]
        )

        ax.annotate(
            rf"$\Delta\phi_O"
            rf"({correct_results['t_opt']})"
            rf"={difference[index]:.3f}$",
            xy=(
                correct_results["t_opt"],
                oracle_modified[index]
            ),
            xytext=(
                correct_results["t_opt"] + 0.35,
                oracle_modified[index] - 0.12
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                linewidth=1.4
            ),
            fontsize=10,
            color="red"
        )

    ax.set_xlabel(
        r"Step $t$",
        fontsize=12
    )

    ax.set_ylabel(
        r"Oracle contribution $\phi_O(t)$",
        fontsize=12
    )

    ax.set_title(
        "Oracle Phase Anomaly: Profile Comparison",
        fontsize=14
    )

    ax.set_xticks(t)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best",
        fontsize=9,
        framealpha=0.95
    )

    plt.tight_layout()

    if save:

        png = f"{filename_prefix}.png"

        pdf = f"{filename_prefix}.pdf"

        fig.savefig(
            png,
            dpi=600,
            bbox_inches="tight"
        )

        fig.savefig(
            pdf,
            bbox_inches="tight"
        )

        print(
            f"Files saved: {png}, {pdf}"
        )

    plt.show()

    plt.close(fig)
