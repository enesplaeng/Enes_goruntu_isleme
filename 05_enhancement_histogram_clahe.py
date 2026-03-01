import cv2
import matplotlib.pyplot as plt
import numpy as np

gray    = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)
hist_eq = cv2.equalizeHist(gray)
clips   = [1.0, 2.0, 4.0]
gorseller = [gray, hist_eq] + [
    cv2.createCLAHE(clipLimit=c, tileGridSize=(8,8)).apply(gray)
    for c in clips
]
basliklar = [
    "Orijinal",
    "Histogram\nEşitleme (Global)",
    "CLAHE\nclip=1.0  (az)",
    "CLAHE\nclip=2.0  (orta)",
    "CLAHE\nclip=4.0  (güçlü)",
]
renkler = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#66bb6a", "#2e7d32"]

fig, axes = plt.subplots(2, 5, figsize=(24, 9))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    "05 — Kontrast İyileştirme  •  Histogram Eşitleme  vs  CLAHE\n"
    "CLAHE = Contrast Limited Adaptive Histogram Equalization  (8×8 döşeme)",
    fontsize=17, fontweight="bold", color="white", y=0.99)

x = np.arange(256)

for i, (gorsel, baslik, brenk) in enumerate(zip(gorseller, basliklar, renkler)):
    std = gorsel.std()
    ort = gorsel.mean()

    # Üst satır: görüntüler
    axes[0, i].imshow(gorsel, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(baslik, fontsize=13, fontweight="bold", color=brenk, pad=8)
    axes[0, i].axis("off")
    axes[0, i].set_facecolor("#0d1117")
    axes[0, i].text(0.5, -0.04, f"Kontrast (Std): {std:.1f}   Ort: {ort:.1f}",
                    transform=axes[0, i].transAxes,
                    ha="center", fontsize=10, color="#b0bec5")

    # Alt satır: histogram
    hist = cv2.calcHist([gorsel], [0], None, [256], [0, 256]).flatten()
    axes[1, i].set_facecolor("#0d1117")
    axes[1, i].fill_between(x, hist, alpha=0.70, color=brenk)
    axes[1, i].plot(x, hist, color=brenk, linewidth=1.2)
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_xlabel("Piksel değeri (0–255)", fontsize=10, color="#b0bec5")
    if i == 0:
        axes[1, i].set_ylabel("Piksel sayısı", fontsize=10, color="#b0bec5")
    axes[1, i].tick_params(colors="#b0bec5", labelsize=9)
    axes[1, i].axhline(hist.mean(), color="yellow", linewidth=1,
                       linestyle="--", alpha=0.6, label=f"Ort={hist.mean():.0f}")
    axes[1, i].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for sp in axes[1, i].spines.values():
        sp.set_edgecolor("#444")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
