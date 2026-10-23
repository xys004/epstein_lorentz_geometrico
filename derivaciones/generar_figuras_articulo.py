"""Genera las figuras vectoriales del primer borrador del artículo.

Las cantidades están normalizadas con c=1 y con la escala geométrica indicada
en cada figura. Los PDF son los originales para LaTeX; los PNG facilitan la
inspección rápida.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.8,
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavuserif",
    }
)

BLUE = "#176B87"
ORANGE = "#D97706"
GREEN = "#2F855A"
PURPLE = "#7C3AED"
GRAY = "#5B6573"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def add_arrow(ax: plt.Axes, x: np.ndarray, y: np.ndarray, index: int) -> None:
    ax.annotate(
        "",
        xy=(x[index + 1], y[index + 1]),
        xytext=(x[index - 1], y[index - 1]),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.0},
    )


def figure_minkowski_vs_spt() -> None:
    u = np.linspace(-1.8, 1.8, 600)
    x = np.cosh(u) - 1.0
    minkowski_time = np.sinh(u)
    proper_time = u

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    axes[0].plot(x, minkowski_time, color=BLUE)
    axes[0].axvline(0.0, color="#D7DCE2", lw=0.8)
    axes[0].axhline(0.0, color="#D7DCE2", lw=0.8)
    axes[0].set(
        xlabel=r"$x/R$",
        ylabel=r"$ct/R$",
        title=r"(a) Minkowski worldline: hyperbola",
    )
    add_arrow(axes[0], x, minkowski_time, 470)

    axes[1].plot(x, proper_time, color=ORANGE)
    axes[1].axvline(0.0, color="#D7DCE2", lw=0.8)
    axes[1].axhline(0.0, color="#D7DCE2", lw=0.8)
    axes[1].set(
        xlabel=r"$x/R$",
        ylabel=r"$c\tau/R$",
        title=r"(b) SPT image: catenary",
    )
    add_arrow(axes[1], x, proper_time, 470)

    for ax in axes:
        ax.grid(color="#E6E9ED", lw=0.55)
        ax.set_xlim(-0.08, 2.2)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("auto")

    fig.suptitle(r"Constant signed proper acceleration, $R=c^2/A_0$", y=1.02)
    fig.tight_layout()
    save(fig, "fig01_hyperbola_catenary")


def colored_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    norm: Normalize,
) -> LineCollection:
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=3.0,
    )
    collection.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def figure_catenary_vs_circle() -> None:
    norm = Normalize(vmin=1.0, vmax=10.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    u = np.linspace(-1.8, 1.8, 600)
    x_cat = np.cosh(u) - 1.0
    y_cat = u
    invariant_cat = np.ones_like(u)
    collection = colored_curve(
        axes[0], x_cat, y_cat, invariant_cat, norm
    )
    axes[0].set(
        xlabel=r"$x/R$",
        ylabel=r"$c\tau/R$",
        title=r"(a) Catenary: $\kappa_E$ varies, $AR/c^2=1$",
    )

    theta = np.linspace(-1.25, 1.25, 600)
    x_circle = 1.0 - np.cos(theta)
    y_circle = np.sin(theta)
    invariant_circle = 1.0 / np.cos(theta) ** 2
    colored_curve(
        axes[1], x_circle, y_circle, invariant_circle, norm
    )
    axes[1].set(
        xlabel=r"$x/L$",
        ylabel=r"$c\tau/L$",
        title=r"(b) Circle: $\kappa_E L=1$, $AL/c^2=\sec^2\theta$",
    )

    for ax in axes:
        ax.grid(color="#E6E9ED", lw=0.55)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.08)

    colorbar = fig.colorbar(
        collection,
        ax=axes,
        location="right",
        fraction=0.04,
        pad=0.04,
    )
    colorbar.set_label(r"normalized proper acceleration")
    fig.suptitle(
        r"Euclidean curvature alone is not the dynamical invariant", y=1.02
    )
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.17, top=0.82, wspace=0.34)
    save(fig, "fig02_catenary_circle")


def figure_uniform_circular_helix() -> None:
    beta = 0.75
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    theta = np.linspace(0.0, 4.5 * np.pi, 900)
    x = np.cos(theta)
    z = np.sin(theta)
    y = theta / (beta * gamma)

    fig = plt.figure(figsize=(6.0, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, z, y, color=GREEN, lw=2.0)
    ax.scatter([x[0]], [z[0]], [y[0]], color=ORANGE, s=22, zorder=5)
    ax.text(x[0], z[0], y[0], "  start", fontsize=8)
    ax.set(
        xlabel=r"$x/R$",
        ylabel=r"$z/R$",
        zlabel=r"$c\tau/R$",
        title=(
            r"Uniform circular motion as a Euclidean SPT helix "
            r"($\beta=0.75$)"
        ),
    )
    # Coarser horizontal ticks keep the projected x/R and z/R labels distinct.
    ax.set_xticks(np.linspace(-1.0, 1.0, 5))
    ax.set_yticks(np.linspace(-1.0, 1.0, 5))
    ax.view_init(elev=22, azim=-54)
    ax.set_box_aspect((1.0, 1.0, 1.35))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.87, 0.89, 0.91, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.55
    fig.tight_layout()
    save(fig, "fig03_uniform_circular_helix")


def figure_boosted_catenaries() -> None:
    u = np.linspace(-2.0, 2.0, 700)
    boosts = (-0.8, 0.0, 0.8)
    colors = (PURPLE, BLUE, ORANGE)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for boost, color in zip(boosts, colors):
        x_prime = np.cosh(u - boost) - np.cosh(boost)
        y = u
        q_prime = 1.0 / np.cosh(u - boost)
        curvature_prime = q_prime**2
        axes[0].plot(
            x_prime,
            y,
            color=color,
            label=rf"$W={boost:+.1f}$",
        )
        axes[1].plot(
            u,
            curvature_prime / q_prime**2,
            color=color,
            label=rf"$W={boost:+.1f}$",
        )

    axes[0].set(
        xlabel=r"$x'/R$",
        ylabel=r"$c\tau/R$",
        title=r"(a) Observer-dependent Euclidean images",
    )
    axes[1].set(
        xlabel=r"$u=A_0\tau/c$",
        ylabel=r"$R\kappa'_E/(q')^2$",
        title=r"(b) Observer-independent weighted curvature",
        ylim=(0.94, 1.06),
    )
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(color="#E6E9ED", lw=0.55)
    fig.suptitle(r"The induced action of three Lorentz boosts", y=1.02)
    fig.tight_layout()
    save(fig, "fig04_boosted_catenaries")


def figure_velocity_geometry() -> None:
    """Open and closed indicatrices in Klein velocity coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    boundary_angle = np.linspace(0.0, 2.0 * np.pi, 700)
    boundary_x = np.cos(boundary_angle)
    boundary_y = np.sin(boundary_angle)

    for ax in axes:
        ax.plot(
            boundary_x,
            boundary_y,
            color="#9AA3AE",
            lw=1.0,
            ls="--",
        )
        ax.fill(boundary_x, boundary_y, color="#F8FAFC", zorder=-5)
        ax.axhline(0.0, color="#E0E5EA", lw=0.6)
        ax.axvline(0.0, color="#E0E5EA", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel(r"$\beta_1$")
        ax.set_ylabel(r"$\beta_2$")
        ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))

    start = np.array([-0.62, -0.28])
    finish = np.array([0.66, 0.34])
    straight_t = np.linspace(0.0, 1.0, 300)
    geodesic = start[:, None] * (1.0 - straight_t) + finish[:, None] * straight_t
    curve = geodesic.copy()
    normal = np.array([-0.42, 0.86])
    curve += normal[:, None] * (0.34 * np.sin(np.pi * straight_t))

    axes[0].plot(geodesic[0], geodesic[1], color=BLUE, lw=2.1)
    axes[0].plot(curve[0], curve[1], color=ORANGE, lw=2.0)
    axes[0].scatter(
        [start[0], finish[0]],
        [start[1], finish[1]],
        color=GRAY,
        s=22,
        zorder=5,
    )
    axes[0].text(-0.04, -0.09, r"$d_h(T_0,T_1)$", color=BLUE, fontsize=8)
    axes[0].text(-0.28, 0.55, r"$L_h[T]$", color=ORANGE, fontsize=8)
    axes[0].set_title(r"(a) Open bound: $L_h[T]\geq d_h(T_0,T_1)$")

    beta_radius = 0.66
    loop_x = beta_radius * np.cos(boundary_angle)
    loop_y = beta_radius * np.sin(boundary_angle)
    axes[1].fill(loop_x, loop_y, color="#DDF3E7", alpha=0.95)
    axes[1].plot(loop_x, loop_y, color=GREEN, lw=2.2)
    add_arrow(axes[1], loop_x, loop_y, 110)
    axes[1].text(
        -0.21,
        0.02,
        r"$\Omega=|\widetilde{\Theta}_T|$",
        color=GREEN,
    )
    axes[1].text(-0.25, -0.78, r"$\mathcal{B}=L_h(C)$", color=GRAY)
    axes[1].set_title(r"(b) Closed bound: hyperbolic circle")

    fig.suptitle(r"Canonical hyperbolic geometry of the SPT tangent", y=1.02)
    fig.tight_layout()
    save(fig, "fig05_velocity_geometry")


def main() -> None:
    figure_minkowski_vs_spt()
    figure_catenary_vs_circle()
    figure_uniform_circular_helix()
    figure_boosted_catenaries()
    figure_velocity_geometry()
    print(f"OK: 5 figuras PDF+PNG generadas en {OUT}")


if __name__ == "__main__":
    main()
