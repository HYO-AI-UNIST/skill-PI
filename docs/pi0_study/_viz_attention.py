"""Pi0 'shared attention' 개념 시각화. (설명용 그림 생성 스크립트)"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#4C72B0"
RED = "#C44E52"
GRAY = "#999999"
LBLUE = "#A6C0E0"
LRED = "#E8B4B6"


def token_box(ax, x, y, w, h, color, label, fs=9, tc="white"):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2, edgecolor="white", facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4, weight="bold")


def arrow(ax, p1, p2, color=GRAY, lw=1.3, style="-|>", rad=0.0, alpha=1.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}", zorder=2, alpha=alpha,
    )
    ax.add_patch(a)


# =====================================================================================
# FIGURE 1 : 따로 vs 공유  (Separate vs Shared attention)
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- 왼쪽: 보통 = 따로 attention ----
ax = axes[0]
ax.set_title("Normal: two SEPARATE transformers\n(vision and motor never talk)", fontsize=13, weight="bold")
# vision pool
for i in range(4):
    token_box(ax, 0.5 + i * 1.0, 4.2, 0.85, 0.7, BLUE, f"img{i+1}")
ax.text(2.4, 5.25, "Vision expert", ha="center", fontsize=11, weight="bold", color=BLUE)
# within-vision arrows
for i in range(4):
    for j in range(4):
        if i != j:
            arrow(ax, (0.92 + i, 4.55), (0.92 + j, 4.55), color=LBLUE, lw=0.7, rad=0.3, alpha=0.7)
# action pool
for i in range(3):
    token_box(ax, 1.0 + i * 1.0, 1.2, 0.85, 0.7, RED, f"act{i+1}")
ax.text(2.4, 0.55, "Motor expert", ha="center", fontsize=11, weight="bold", color=RED)
for i in range(3):
    for j in range(3):
        if i != j:
            arrow(ax, (1.42 + i, 1.55), (1.42 + j, 1.55), color=LRED, lw=0.7, rad=0.3, alpha=0.7)
# big X between them
ax.text(2.4, 3.0, "no connection", ha="center", fontsize=12, color="black", style="italic")
ax.plot([1.7, 3.1], [2.7, 3.3], color="black", lw=2.5)
ax.plot([1.7, 3.1], [3.3, 2.7], color="black", lw=2.5)
ax.set_xlim(0, 5)
ax.set_ylim(0, 6)
ax.axis("off")

# ---- 오른쪽: Pi0 = 공유 attention ----
ax = axes[1]
ax.set_title("Pi0: ONE shared attention pool\n(motor token can look at any image token)", fontsize=13, weight="bold")
labels = [("img1", BLUE), ("img2", BLUE), ("img3", BLUE), ("img4", BLUE), ("act1", RED), ("act2", RED), ("act3", RED)]
xs = []
for i, (lab, col) in enumerate(labels):
    x = 0.4 + i * 1.05
    xs.append(x + 0.42)
    token_box(ax, x, 2.9, 0.85, 0.8, col, lab)
ax.text(2.5, 4.05, "PaliGemma (vision+lang)", ha="center", fontsize=10.5, weight="bold", color=BLUE)
ax.text(6.1, 4.05, "Action expert", ha="center", fontsize=10.5, weight="bold", color=RED)
ax.plot([0.2, 4.65], [3.9, 3.9], color=BLUE, lw=2)
ax.plot([4.75, 7.2], [3.9, 3.9], color=RED, lw=2)
# action tokens reach back to ALL tokens (the key point)
for ai in [4, 5, 6]:
    for tj in range(7):
        if ai != tj:
            c = RED if tj >= 4 else "#7C9FD0"
            arrow(ax, (xs[ai], 2.85), (xs[tj], 2.85), color=c, lw=0.8, rad=-0.45, alpha=0.55)
ax.text(2.5, 1.2, "KEY IDEA: action tokens (red) send their 'query' down\nand read image/language tokens (blue) in the SAME attention.",
        ha="center", fontsize=11, color="black")
ax.text(2.5, 0.45, "-> motor decisions become conditioned on what the camera sees.",
        ha="center", fontsize=11, color=RED, weight="bold")
ax.set_xlim(0, 7.6)
ax.set_ylim(0, 6)
ax.axis("off")

plt.tight_layout()
plt.savefig("/workspace/AI_dev/phyisicial_intelligence/skill-based_PI0/pi0_attention_concept.png", dpi=130, bbox_inches="tight")
plt.close()


# =====================================================================================
# FIGURE 2 : 어떻게 공유가 가능한가 (mechanics: project to common Q/K/V space)
# =====================================================================================
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_title("HOW sharing works: different widths -> SAME q/k/v shape -> one attention -> split back",
             fontsize=13, weight="bold")

# prefix tokens (2048d) left-top, suffix tokens (1024d) left-bottom
token_box(ax, 0.3, 5.6, 2.0, 0.9, BLUE, "prefix tokens\n(dim 2048)", fs=10)
token_box(ax, 0.3, 1.6, 2.0, 0.9, RED, "suffix tokens\n(dim 1024)", fs=10)

# own projections
token_box(ax, 3.2, 5.6, 2.3, 0.9, "#6B8CBE", "PaliGemma\nWq/Wk/Wv", fs=9.5)
token_box(ax, 3.2, 1.6, 2.3, 0.9, "#D17A7D", "ActionExpert\nWq/Wk/Wv", fs=9.5)
arrow(ax, (2.3, 6.05), (3.2, 6.05), lw=1.6)
arrow(ax, (2.3, 2.05), (3.2, 2.05), lw=1.6)

# common q/k/v (same head_dim)
token_box(ax, 6.5, 5.6, 2.2, 0.9, "#3B5A82", "Q,K,V\nhead_dim=256", fs=9.5)
token_box(ax, 6.5, 1.6, 2.2, 0.9, "#9C3B3E", "Q,K,V\nhead_dim=256", fs=9.5)
arrow(ax, (5.5, 6.05), (6.5, 6.05), lw=1.6)
arrow(ax, (5.5, 2.05), (6.5, 2.05), lw=1.6)
ax.text(7.6, 4.7, "same shape!", ha="center", fontsize=10.5, color="black", style="italic", weight="bold")

# concat into one attention
token_box(ax, 9.4, 3.4, 2.4, 1.5, "#5A4A6A", "CONCAT\n+ one\nattention", fs=11)
arrow(ax, (8.7, 6.05), (9.4, 4.6), color=BLUE, lw=1.8, rad=-0.2)
arrow(ax, (8.7, 2.05), (9.4, 3.7), color=RED, lw=1.8, rad=0.2)

# split back to own out-projection / own dims
token_box(ax, 12.3, 5.6, 1.9, 0.9, BLUE, "prefix out\n(dim 2048)", fs=9.5)
token_box(ax, 12.3, 1.6, 1.9, 0.9, RED, "suffix out\n(dim 1024)", fs=9.5)
arrow(ax, (11.8, 4.5), (12.3, 6.05), color=BLUE, lw=1.8, rad=-0.2)
arrow(ax, (11.8, 3.8), (12.3, 2.05), color=RED, lw=1.8, rad=0.2)

ax.text(7.1, 0.4,
        "Each expert keeps its OWN weights (blue vs red), but both translate into a common q/k/v space\n"
        "(same head_dim & #heads). So all tokens mix in ONE attention, then each expert reads its slice back.",
        ha="center", fontsize=10.8, color="black")
ax.set_xlim(0, 14.6)
ax.set_ylim(0, 7.2)
ax.axis("off")
plt.tight_layout()
plt.savefig("/workspace/AI_dev/phyisicial_intelligence/skill-based_PI0/pi0_attention_mechanics.png", dpi=130, bbox_inches="tight")
plt.close()


# =====================================================================================
# FIGURE 3 : attention mask heatmap (누가 누구를 보나)
# =====================================================================================
# toy 시퀀스: img x6, lang x3, state x1, action x4  (faithful to make_attn_mask)
n_img, n_lang, n_state, n_act = 6, 3, 1, 4
ar = [0] * n_img + [0] * n_lang + [1] + ([1] + [0] * (n_act - 1))
ar = np.array(ar)
cum = np.cumsum(ar)
N = len(ar)
# mask[q, k] = (cum[k] <= cum[q])
mask = (cum[None, :] <= cum[:, None]).astype(float)

labels = (
    [f"img{i+1}" for i in range(n_img)]
    + [f"lng{i+1}" for i in range(n_lang)]
    + ["state"]
    + [f"act{i+1}" for i in range(n_act)]
)

fig, ax = plt.subplots(figsize=(9.5, 8.5))
ax.imshow(mask, cmap="Greens", vmin=0, vmax=1.4)
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("KEY  (token being looked AT)", fontsize=12, weight="bold")
ax.set_ylabel("QUERY  (token doing the looking)", fontsize=12, weight="bold")
ax.set_title("Pi0 attention mask:  green = 'query row CAN see key column'", fontsize=12.5, weight="bold")
for i in range(N):
    for j in range(N):
        ax.text(j, i, "O" if mask[i, j] else ".", ha="center", va="center",
                fontsize=8, color=("white" if mask[i, j] else "#bbbbbb"))
# block separators
for b in [n_img - 0.5, n_img + n_lang - 0.5, n_img + n_lang + n_state - 0.5]:
    ax.axhline(b, color="black", lw=1.6)
    ax.axvline(b, color="black", lw=1.6)
# annotate
ax.text(1.0, n_img + n_lang + n_state + 1.5, "actions see\nEVERYTHING", color=RED, fontsize=10, weight="bold", ha="center")
ax.text(n_img + n_lang + 1.7, 1.5, "prefix (img+lang)\nCANNOT see\nstate/actions",
        color=BLUE, fontsize=9.5, weight="bold", ha="center", va="center")
plt.tight_layout()
plt.savefig("/workspace/AI_dev/phyisicial_intelligence/skill-based_PI0/pi0_attention_mask.png", dpi=130, bbox_inches="tight")
plt.close()

print("saved 3 figures.")
print("mask matrix (rows=query, cols=key), O=can attend:")
print("labels:", labels)
for i in range(N):
    print(labels[i].rjust(6), " ".join("O" if mask[i, j] else "." for j in range(N)))
