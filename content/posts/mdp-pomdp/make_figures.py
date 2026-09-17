# -*- coding: utf-8 -*-
"""Figures for the Markov models / reinforcement learning post.

- markov-model-family.png       cover: MC -> {MDP, HMM}; MDP + HMM -> POMDP
- markov-chain-vs-hmm.png       the same two-state chain, seen vs hidden
- agent-environment-loop.png    the agent-environment interaction loop
- mdp-graphical-model.png       MDP as a graphical model over time
- pomdp-graphical-model.png     POMDP as a graphical model over time

Every panel label is language-neutral (MC / MDP / HMM / POMDP, node symbols,
Latin tag words), so a single PNG serves both the Chinese and the English post.
All axes that draw circles or diamonds use equal aspect, so they stay round.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Polygon)

font_manager.fontManager.addfont(r"C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"

TINT_BLUE = "#dbe7f7"
TINT_ORANGE = "#fbe4d8"
TINT_AQUA = "#d9f1e7"
TINT_YELLOW = "#f7ecd0"
TINT_VIOLET = "#e4e1f2"
TINT_GRAY = "#f0efec"

EDGE_BLUE = "#2a78d6"
EDGE_ORANGE = "#eb6834"
EDGE_AQUA = "#1baf7a"
EDGE_YELLOW = "#eda100"
EDGE_VIOLET = "#4a3aa7"
EDGE_GRAY = "#898781"

DARK_ORANGE = "#b8461c"
DARK_AQUA = "#12805a"


def blank(figsize, xlim=(0, 1), ylim=(0, 1), equal=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, cx, cy, w, h, fill, edge, lw=1.8, z=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fill, edgecolor=edge,
                                linewidth=lw, zorder=z))


def node(ax, x, y, r, label, fill, edge, fs=13, lw=1.7, dashed=False, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                        linewidth=lw, zorder=z,
                        linestyle="dashed" if dashed else "solid"))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            color=INK if not dashed else INK2, zorder=z + 1)


def arrow(ax, p0, p1, color, rad=0.0, lw=1.7, ls="solid", z=2, scale=15):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=scale,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def path(ax, pts, color, lw=1.7, ls="solid", z=2, scale=15):
    """Polyline whose final segment carries the arrow head."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle=ls, zorder=z, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color, lw=lw, ls=ls, z=z, scale=scale)


def self_loop(ax, cx, cy, r, color, lw=1.6, fs=12, label=None, z=2):
    """A visible loop leaving and re-entering the top of a node."""
    loop_c = np.array([cx, cy + 1.152 * r])
    R = 0.75 * r
    th = np.radians(np.linspace(211, -31, 90))
    xs = loop_c[0] + R * np.cos(th)
    ys = loop_c[1] + R * np.sin(th)
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=z,
            solid_capstyle="round")
    arrow(ax, (xs[-3], ys[-3]), (xs[-1], ys[-1]), color, lw=lw, z=z,
          scale=12)
    if label:
        ax.text(cx, cy + 1.152 * r + R + 0.075 * r, label, ha="center",
                va="bottom", fontsize=fs, color=INK, zorder=z + 1)


# ===================================================================== cover
fig, ax = blank((10, 6.4))

# the reinforcement-learning region: MDP and POMDP live inside, HMM outside
ax.add_patch(Ellipse((0.38, 0.45), 0.78, 0.84, facecolor=TINT_ORANGE,
                     edgecolor=EDGE_ORANGE, linewidth=2.0, zorder=1))
ax.text(0.110, 0.430, "Reinforcement", ha="center", va="center",
        fontsize=15, color=DARK_ORANGE, zorder=2)
ax.text(0.110, 0.370, "Learning", ha="center", va="center",
        fontsize=15, color=DARK_ORANGE, zorder=2)

rbox(ax, 0.60, 0.90, 0.27, 0.135, TINT_AQUA, EDGE_AQUA)
ax.text(0.60, 0.90, "Markov Chain", ha="center", va="center",
        fontsize=15, color=INK, zorder=3)

rbox(ax, 0.36, 0.62, 0.32, 0.165, TINT_BLUE, EDGE_BLUE)
ax.text(0.36, 0.645, "MDP", ha="center", va="center",
        fontsize=15, color=INK, zorder=3)
ax.text(0.36, 0.570, "Markov Decision Process", ha="center", va="center",
        fontsize=10, color=INK2, zorder=3)

rbox(ax, 0.878, 0.575, 0.235, 0.165, TINT_VIOLET, EDGE_VIOLET)
ax.text(0.878, 0.600, "HMM", ha="center", va="center",
        fontsize=15, color=INK, zorder=3)
ax.text(0.878, 0.525, "Hidden Markov Model", ha="center", va="center",
        fontsize=10, color=INK2, zorder=3)

rbox(ax, 0.47, 0.21, 0.30, 0.16, TINT_YELLOW, EDGE_YELLOW)
ax.text(0.47, 0.235, "POMDP", ha="center", va="center",
        fontsize=15, color=INK, zorder=3)
ax.text(0.47, 0.162, "Partially Observable MDP", ha="center", va="center",
        fontsize=10, color=INK2, zorder=3)

# Markov chain -> MDP / HMM
arrow(ax, (0.500, 0.845), (0.378, 0.710), INK2, lw=2.0, scale=18)
arrow(ax, (0.698, 0.845), (0.816, 0.662), INK2, lw=2.0, scale=18)

# MDP + HMM -> POMDP
ax.text(0.47, 0.425, "+", ha="center", va="center", fontsize=30,
        color=EDGE_ORANGE, zorder=3)
arrow(ax, (0.47, 0.365), (0.47, 0.298), EDGE_ORANGE, lw=2.4, scale=20)

fig.savefig("markov-model-family.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ======================================================== Markov chain vs HMM
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(wspace=0.05)

R = 0.10
C1, C2, CY = 0.30, 0.74, 0.66


def two_state_chain(ax, hidden):
    """Draw S1 <-> S2 with self loops; hidden states render as dashed."""
    fill = TINT_GRAY if hidden else TINT_BLUE
    edge = EDGE_GRAY if hidden else EDGE_BLUE
    node(ax, C1, CY, R, "$S_1$", fill, edge, dashed=hidden)
    node(ax, C2, CY, R, "$S_2$", fill, edge, dashed=hidden)
    # S1 -> S2 above, S2 -> S1 below
    arrow(ax, (C1 + 0.062, CY + 0.062), (C2 - 0.062, CY + 0.062), EDGE_AQUA)
    ax.text((C1 + C2) / 2, CY + 0.105, "0.9", ha="center", va="bottom",
            fontsize=12, color=INK)
    arrow(ax, (C2 - 0.062, CY - 0.062), (C1 + 0.062, CY - 0.062), EDGE_AQUA)
    ax.text((C1 + C2) / 2, CY - 0.105, "0.8", ha="center", va="top",
            fontsize=12, color=INK)
    self_loop(ax, C1, CY, R, EDGE_AQUA, label="0.1")
    self_loop(ax, C2, CY, R, EDGE_AQUA, label="0.2")


for ax, title in zip(axes, ["(a) Markov chain (MC)",
                            "(b) Hidden Markov model (HMM)"]):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=15, color=INK, pad=14)

ax = axes[0]
two_state_chain(ax, hidden=False)
ax.text(0.52, 0.20, "the state itself is observed", ha="center", va="center",
        fontsize=13, color=INK2)

ax = axes[1]
two_state_chain(ax, hidden=True)
node(ax, C1, 0.20, R, "$O_1$", TINT_VIOLET, EDGE_VIOLET)
node(ax, C2, 0.20, R, "$O_2$", TINT_VIOLET, EDGE_VIOLET)
arrow(ax, (C1, CY - R), (C1, 0.20 + R), EDGE_VIOLET, ls="dashed")
arrow(ax, (C2, CY - R), (C2, 0.20 + R), EDGE_VIOLET, ls="dashed")
ax.text(C1 - 0.028, (CY + 0.20) / 2, "0.75", ha="right",
        va="center", fontsize=12, color=EDGE_VIOLET)
ax.text(C2 + 0.028, (CY + 0.20) / 2, "0.75", ha="left",
        va="center", fontsize=12, color=EDGE_VIOLET)

fig.savefig("markov-chain-vs-hmm.png", facecolor=SURFACE,
            bbox_inches="tight")
plt.close(fig)

# ====================================================== agent-environment loop
fig, ax = blank((7.6, 3.9))

rbox(ax, 0.42, 0.72, 0.34, 0.26, TINT_BLUE, EDGE_BLUE)
ax.text(0.42, 0.72, "Agent", ha="center", va="center", fontsize=17,
        color=INK, zorder=3)

rbox(ax, 0.42, 0.26, 0.34, 0.26, TINT_GRAY, EDGE_GRAY)
ax.text(0.42, 0.26, "Environment", ha="center", va="center", fontsize=17,
        color=INK, zorder=3)

# action: agent -> environment, down the right-hand side
path(ax, [(0.59, 0.72), (0.88, 0.72), (0.88, 0.26), (0.59, 0.26)],
     EDGE_ORANGE, lw=2.0, scale=18)
ax.text(0.945, 0.49, "action  $a_t$", ha="center", va="center", fontsize=14,
        color=DARK_ORANGE, rotation=-90)

# state and reward: environment -> agent, down the left-hand side
path(ax, [(0.25, 0.26), (0.10, 0.26), (0.10, 0.72), (0.25, 0.72)],
     EDGE_AQUA, lw=2.0, scale=18)
ax.text(0.045, 0.49, "$s_{t+1}$,  $r_t$", ha="center", va="center",
        fontsize=14, color=DARK_AQUA, rotation=-90)

fig.savefig("agent-environment-loop.png", facecolor=SURFACE,
            bbox_inches="tight")
plt.close(fig)


# ================================================== graphical model helpers
def diamond(ax, cx, cy, w, h, fill, edge, label, fs=13):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, facecolor=fill, edgecolor=edge,
                         linewidth=1.7, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)


def square(ax, cx, cy, s, fill, edge, label, fs=13):
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                                boxstyle="round,pad=0.002,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.7, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)


# ============================================================ MDP graph model
# xlim 0..2.4 over a 9.6 x 4.0 figure keeps the aspect at 1:1
fig, ax = blank((9.6, 4.0), xlim=(0, 2.4), equal=True)
XS = [0.72, 1.20, 1.68]
RS = 0.13          # state radius
for i, x in enumerate(XS):
    node(ax, x, 0.16, RS, f"$S_{i}$", TINT_BLUE, EDGE_BLUE)
    diamond(ax, x, 0.50, 0.30, 0.24, TINT_YELLOW, EDGE_YELLOW, f"$R_{i}$")
    square(ax, x, 0.84, 0.26, TINT_ORANGE, EDGE_ORANGE, f"$A_{i}$")
    # S -> R
    arrow(ax, (x, 0.16 + RS), (x, 0.38), INK2, scale=13)
    # A -> R
    arrow(ax, (x - 0.07, 0.71), (x - 0.07, 0.568), INK2, scale=13)
    # S -> A, routed around the left of R. It leaves the state circle at its
    # upper-left and runs at y = 0.252, clear of the S -> S' arrows at y = 0.16.
    path(ax, [(x - 0.092, 0.252), (x - 0.32, 0.252), (x - 0.32, 0.84),
              (x - 0.13, 0.84)], INK2, scale=13)
for x0, x1 in zip(XS[:-1], XS[1:]):
    arrow(ax, (x0 + RS, 0.16), (x1 - RS, 0.16), INK2, scale=13)

for y, name, col in [(0.16, "state", EDGE_BLUE), (0.50, "reward",
                                                  EDGE_YELLOW),
                     (0.84, "action", EDGE_ORANGE)]:
    ax.text(0.30, y, name, ha="right", va="center", fontsize=13, color=col)
ax.text(2.34, 0.98, "time  $t \\rightarrow$", ha="right", va="top",
        fontsize=13, color=INK2)

fig.savefig("mdp-graphical-model.png", facecolor=SURFACE,
            bbox_inches="tight")
plt.close(fig)

# ========================================================== POMDP graph model
# xlim 0..1.60 over a 9.0 x 5.625 figure keeps the aspect at 1:1
fig, ax = blank((9.0, 5.625), xlim=(0, 1.60), equal=True)
SX, SY, SR = 0.52, 0.35, 0.105
S2X = 1.15

node(ax, SX, SY, SR, "$S$", TINT_BLUE, EDGE_BLUE, fs=14)
node(ax, S2X, SY, SR, "$S'$", TINT_BLUE, EDGE_BLUE, fs=14)
diamond(ax, SX, 0.63, 0.26, 0.20, TINT_YELLOW, EDGE_YELLOW, "$R$")
square(ax, SX, 0.88, 0.20, TINT_ORANGE, EDGE_ORANGE, "$A$", fs=12)
node(ax, SX, 0.095, 0.09, "$O$", TINT_VIOLET, EDGE_VIOLET, fs=13)

# S -> R and A -> R
arrow(ax, (SX, SY + SR), (SX, 0.53), INK2, scale=13)
arrow(ax, (SX - 0.05, 0.78), (SX - 0.05, 0.695), INK2, scale=13)
# S -> A, routed around the left of R. Leaves the state circle at its
# upper-left and runs at y = 0.424, clear of the S -> S' arrow at y = 0.35.
path(ax, [(SX - 0.074, 0.424), (0.24, 0.424), (0.24, 0.88), (SX - 0.10, 0.88)],
     INK2, scale=13)
# S -> S'
arrow(ax, (SX + SR, SY), (S2X - SR, SY), INK2, scale=13)
# A -> S' : action influences the next state
arrow(ax, (SX + 0.10, 0.88), (1.076, 0.424), INK2, scale=13)
# S -> O : the sensor model, the edge that makes this a POMDP
arrow(ax, (SX, SY - SR), (SX, 0.185), EDGE_VIOLET, lw=2.1, scale=13)

ax.text(0.66, 0.20, "sensor model", ha="left", va="center", fontsize=12,
        color=EDGE_VIOLET)
ax.text(0.66, 0.075, "decisions see $O$, not $S$", ha="left", va="center",
        fontsize=12, color=INK2)

fig.savefig("pomdp-graphical-model.png", facecolor=SURFACE,
            bbox_inches="tight")
plt.close(fig)


# ================================================= POMDP agent-environment loop
# xlim 0..2.15 over a 12.9 x 6.0 figure keeps the aspect at 1:1


def badge(ax, x, y, n, r=0.042, fs=12, z=6):
    """The circled step number (1)..(4) used by the CS188 interaction figure."""
    ax.add_patch(Circle((x, y), r, facecolor=SURFACE, edgecolor=INK2,
                        linewidth=1.5, zorder=z))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs,
            color=INK2, zorder=z + 1)


def dbox(ax, cx, cy, w, h, label, fs=12):
    """A dashed box holding one sampling statement."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.003,rounding_size=0.018",
                                facecolor=TINT_VIOLET, edgecolor=EDGE_VIOLET,
                                linestyle="dashed", linewidth=1.4, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=EDGE_VIOLET, zorder=5)


fig, ax = blank((12.9, 6.0), xlim=(0, 2.15), equal=True)

# the two boxes: environment on top, agent below
rbox(ax, 1.09, 0.845, 0.80, 0.30, TINT_GRAY, EDGE_GRAY)
ax.text(1.09, 0.945, "Environment", ha="center", va="center", fontsize=15,
        color=INK, zorder=3)
badge(ax, 0.90, 0.862, 2, r=0.038)
ax.text(0.96, 0.862, r"$s \in \mathcal{S} \longrightarrow s' \in \mathcal{S}$",
        ha="left", va="center", fontsize=13, color=INK, zorder=3)
dbox(ax, 1.09, 0.760, 0.50, 0.078, r"$s' \sim T(s' \mid s, a)$")

rbox(ax, 1.09, 0.315, 0.80, 0.20, TINT_BLUE, EDGE_BLUE)
ax.text(1.09, 0.362, "Agent", ha="center", va="center", fontsize=15,
        color=INK, zorder=3)
badge(ax, 0.90, 0.268, 4, r=0.038)
ax.text(0.96, 0.268,
        r"$h_t \rightarrow h_{t+1}, \; b_t \rightarrow b_{t+1}$",
        ha="left", va="center", fontsize=13, color=INK, zorder=3)

# (1) action: agent -> environment, up the left-hand side
path(ax, [(0.69, 0.315), (0.41, 0.315), (0.41, 0.845), (0.69, 0.845)],
     EDGE_ORANGE, lw=2.0, scale=18)
badge(ax, 0.41, 0.580, 1)
ax.text(0.20, 0.680, r"$a \in \mathcal{A}$", ha="center", va="center",
        fontsize=13, color=INK)
dbox(ax, 0.20, 0.575, 0.28, 0.078, r"$a \sim \pi(a \mid h_t)$")

# (3) observation and reward: environment -> agent, down the right-hand side
path(ax, [(1.49, 0.845), (1.77, 0.845), (1.77, 0.315), (1.49, 0.315)],
     EDGE_AQUA, lw=2.0, scale=18)
badge(ax, 1.77, 0.605, 3)
ax.text(1.98, 0.790, r"$o \in \Omega$", ha="center", va="center",
        fontsize=13, color=INK)
dbox(ax, 1.98, 0.665, 0.32, 0.078, r"$o \sim O(o \mid s', a)$")
ax.text(1.98, 0.525, r"$r \in \mathbb{R}$", ha="center", va="center",
        fontsize=13, color=INK)
dbox(ax, 1.98, 0.400, 0.32, 0.078, r"$r \sim R(s, a)$")

# what the agent carries below the belt: history and belief
ax.text(1.09, 0.155,
        r"$h_t = (a_1, o_1, \ldots, a_{t-1}, o_{t-1})$",
        ha="center", va="center", fontsize=13, color=INK2)
ax.text(1.09, 0.072, r"$b_t(s) = \mathrm{Pr}(s \mid h_t)$",
        ha="center", va="center", fontsize=13, color=INK2)

fig.savefig("pomdp-agent-environment.png", facecolor=SURFACE,
            bbox_inches="tight")
plt.close(fig)

print("done")
