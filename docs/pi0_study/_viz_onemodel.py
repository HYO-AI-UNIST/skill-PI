"""하나의 backbone이 텍스트와 action을 둘 다 만드는 원리 시각화."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#4C72B0"; RED = "#C44E52"; GRAY = "#8a8f96"; GREEN = "#3a8f5a"; PURPLE = "#5A4A6A"


def box(ax, x, y, w, h, color, label, fs=10, tc="white", ec="white", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=lw, edgecolor=ec, facecolor=color, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4, weight="bold")


def arr(ax, p1, p2, color=GRAY, lw=1.6, style="-|>", rad=0.0, alpha=1.0, ms=13):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=2, alpha=alpha))


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9.5); ax.axis("off")
ax.set_title("ONE model, two jobs: the backbone only makes 'meaning vectors'.\nOnly the tiny OUTPUT HEAD differs.",
             fontsize=14, weight="bold")

# --- input tokens (bottom) ---
inp = [("img", BLUE), ("img", BLUE), ("Task:..", GRAY), ("State:..", GRAY), ("act", RED), ("act", RED)]
x0 = 3.0
for i, (lab, col) in enumerate(inp):
    box(ax, x0 + i*1.25, 0.5, 1.05, 0.7, col, lab, fs=8.5)
ax.text(x0 + 2.0, 1.45, "image + text prompt tokens", fontsize=9, color="#333", ha="center")
ax.text(x0 + 5.3, 1.45, "action tokens\n(low-level only)", fontsize=8.5, color=RED, ha="center")

# --- shared backbone (middle) ---
box(ax, 2.4, 2.6, 9.2, 1.7, PURPLE,
    "SHARED transformer backbone  (PaliGemma weights)\nturns ANY tokens -> rich 'meaning vectors' (2048-d)", fs=11.5)
for i in range(6):
    arr(ax, (x0 + i*1.25 + 0.52, 1.2), (x0 + i*1.25 + 0.52, 2.6), color="#aaa", lw=1.2, ms=9)

# --- two heads (top) ---
# left: text head
box(ax, 2.0, 5.6, 3.6, 1.1, GREEN, "TEXT head\n(x vocab matrix -> 257k scores)", fs=9.5)
arr(ax, (4.0, 4.3), (3.8, 5.6), color=GREEN, lw=2.0)
box(ax, 2.3, 7.5, 3.0, 0.8, "#2f7a4a", '"pick" "up" "the" "plate"', fs=9.5)
arr(ax, (3.8, 6.7), (3.8, 7.5), color=GREEN, lw=2.0)
# autoregressive loop
arr(ax, (5.3, 7.9), (6.0, 7.9), color=GREEN, lw=1.6, rad=0)
arr(ax, (6.2, 7.6), (6.2, 1.0), color=GREEN, lw=1.6, rad=0.32, style="-|>", alpha=0.8)
ax.text(7.2, 4.6, "autoregressive:\nfeed each word\nback as input,\none at a time",
        fontsize=8.8, color=GREEN, ha="center", weight="bold")

# right: action head
box(ax, 8.6, 5.6, 3.6, 1.1, RED, "action_out_proj\n(2048 -> 32 numbers)", fs=9.5)
arr(ax, (10.2, 4.3), (10.4, 5.6), color=RED, lw=2.0)
box(ax, 8.9, 7.5, 3.0, 0.8, "#9c3b3e", "velocity v_t  (32 floats)", fs=9.5)
arr(ax, (10.4, 6.7), (10.4, 7.5), color=RED, lw=2.0)
# denoising loop
arr(ax, (11.9, 7.9), (12.6, 7.9), color=RED, lw=1.6, rad=0)
arr(ax, (12.8, 7.6), (12.8, 1.0), color=RED, lw=1.6, rad=-0.32, style="-|>", alpha=0.8)
ax.text(13.0, 4.6, "denoising:\nx = x + dt*v_t,\nrepeat 10x",
        fontsize=8.8, color=RED, ha="center", weight="bold")

ax.text(7.0, 9.0 - 8.95 + 0.15 + 0.0, "", fontsize=1)
ax.text(7.0, 0.05, "Same heavy backbone makes the vector. A TINY head decides: discrete word OR continuous number.",
        fontsize=10.5, ha="center", color="#222", weight="bold")
fig.savefig("/workspace/AI_dev/phyisicial_intelligence/skill-based_PI0/pi0_one_model_two_jobs.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("saved")
