"""학습의 joint forward vs 추론의 cache 분리 시각화."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#4C72B0"
RED = "#C44E52"
GRAY = "#9aa0a6"
PURPLE = "#5A4A6A"


def box(ax, x, y, w, h, color, label, fs=8.5, tc="white", ec="white", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=lw, edgecolor=ec, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4, weight="bold")


def arr(ax, p1, p2, color=GRAY, lw=1.3, style="-|>", rad=0.0, alpha=1.0, ms=11):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=2, alpha=alpha))


# 5 토큰: P1 P2 P3 (blue, 2048) | S1 S2 (red, 1024)
TOK = [("P1", BLUE), ("P2", BLUE), ("P3", BLUE), ("S1", RED), ("S2", RED)]


def draw_layer_row(ax, x0, y, tw=0.78, th=0.62, gap=0.18, internal_attn=True):
    """한 레이어 안의 5개 토큰 행을 그리고 중심 x좌표 리스트 반환."""
    xs = []
    for i, (lab, col) in enumerate(TOK):
        x = x0 + i * (tw + gap)
        xs.append(x + tw / 2)
        box(ax, x, y, tw, th, col, lab, fs=8.5)
    return xs, th


fig = plt.figure(figsize=(16, 9.5))

# =====================================================================================
# LEFT PANEL : TRAINING (one joint pass)
# =====================================================================================
ax = fig.add_axes([0.02, 0.05, 0.56, 0.9])
ax.set_title("TRAINING: ONE joint forward pass\n(prefix P + suffix S move together, layer by layer)",
             fontsize=13, weight="bold")
ax.set_xlim(0, 6.4)
ax.set_ylim(0, 11)
ax.axis("off")

x0 = 0.7
layer_ys = [8.7, 6.2, 3.7]  # Layer1, Layer2, Layer3 (top->down)
for li, ly in enumerate(layer_ys):
    # layer container
    ax.add_patch(FancyBboxPatch((x0 - 0.35, ly - 0.55), 5.55, 1.75,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.4, edgecolor="#cccccc", facecolor="#f6f6f8", zorder=1))
    ax.text(x0 - 0.15, ly + 1.0, f"Layer {li+1}", fontsize=10.5, weight="bold", color="#444")
    xs, th = draw_layer_row(ax, x0, ly)
    yc = ly + th / 2
    # shared attention arrows: S1,S2 read all P + each other (curved, above row)
    for si in [3, 4]:
        for tj in range(5):
            if si != tj:
                c = RED if tj >= 3 else "#7C9FD0"
                arr(ax, (xs[si], ly + th), (xs[tj], ly + th), color=c, lw=0.8, rad=-0.4, alpha=0.5, ms=8)
    ax.text(x0 + 4.95, ly + 0.55, "shared\nattention", fontsize=8.2, color=PURPLE, weight="bold", ha="left", va="center")

# vertical flow arrows between layers (all 5 tokens advance together)
for a, b in [(layer_ys[0], layer_ys[1]), (layer_ys[1], layer_ys[2])]:
    for i in range(5):
        x = x0 + i * (0.78 + 0.18) + 0.39
        arr(ax, (x, a - 0.55), (x, b + 1.2), color=GRAY, lw=1.2, rad=0, ms=9)

# input arrows
for i, (lab, col) in enumerate(TOK):
    x = x0 + i * (0.78 + 0.18) + 0.39
    arr(ax, (x, 10.4), (x, layer_ys[0] + 1.2), color=col, lw=1.3, ms=9)
ax.text(x0 + 0.9, 10.55, "prefix tokens (2048d)", fontsize=9, color=BLUE, weight="bold")
ax.text(x0 + 3.3, 10.15, "suffix (1024d)", fontsize=9, color=RED, weight="bold")

# dots for remaining layers + output
ax.text(x0 + 2.4, 3.1, "$\\vdots$  (x depth = 18 layers)", fontsize=12, ha="center", color="#444")
for i in range(5):
    x = x0 + i * (0.78 + 0.18) + 0.39
    arr(ax, (x, 3.7 - 0.55), (x, 1.7), color=GRAY, lw=1.2, ms=9)
# only suffix output used
box(ax, x0 + 3 * (0.78 + 0.18), 0.95, 0.78 * 2 + 0.18, 0.6, RED, "suffix out -> action_out_proj -> v_t", fs=8.2)
ax.text(x0 + 1.3, 1.25, "prefix out\n(not used in loss)", fontsize=8.2, color=BLUE, ha="center", weight="bold")

# =====================================================================================
# RIGHT PANEL : INFERENCE (cache split)
# =====================================================================================
ax = fig.add_axes([0.62, 0.05, 0.36, 0.9])
ax.set_title("INFERENCE: prefix once -> KV cache\n-> suffix reads cache each step",
             fontsize=13, weight="bold")
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 11)
ax.axis("off")

px = 0.5
sx = 3.4
for li, ly in enumerate(layer_ys):
    # prefix layer (left)
    box(ax, px, ly, 1.5, 0.85, BLUE, f"Layer {li+1}\nP1 P2 P3", fs=8.5)
    # cache box (middle)
    box(ax, px + 1.75, ly + 0.05, 0.9, 0.72, GRAY, "KV\ncache", fs=8, tc="white")
    arr(ax, (px + 1.5, ly + 0.42), (px + 1.75, ly + 0.42), color="#777", lw=1.2, ms=9)
    # suffix layer (right) reads cache
    box(ax, sx, ly, 1.2, 0.85, RED, f"Layer {li+1}\nS1 S2", fs=8.5)
    arr(ax, (px + 2.65, ly + 0.42), (sx, ly + 0.42), color="#777", lw=1.2, rad=0.0, ms=9, style="-|>")

# vertical flow prefix
for a, b in [(layer_ys[0], layer_ys[1]), (layer_ys[1], layer_ys[2])]:
    arr(ax, (px + 0.75, a), (px + 0.75, b + 0.85), color=BLUE, lw=1.3, ms=9)
    arr(ax, (sx + 0.6, a), (sx + 0.6, b + 0.85), color=RED, lw=1.3, ms=9)

ax.text(px + 0.75, 10.5, "STEP 1\nprefix all layers\n(once)", fontsize=8.8, ha="center", color=BLUE, weight="bold")
ax.text(sx + 0.6, 10.5, "STEP 2\nsuffix all layers\n(every denoise step)", fontsize=8.8, ha="center", color=RED, weight="bold")
ax.text(2.5, 2.6, "mask blocks prefix->suffix, so prefix\nis independent of suffix = precomputing\n& caching gives the SAME result!", fontsize=9.0, ha="center", color="#333")

fig.savefig("/workspace/AI_dev/phyisicial_intelligence/skill-based_PI0/pi0_joint_vs_cache.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("saved pi0_joint_vs_cache.png")
