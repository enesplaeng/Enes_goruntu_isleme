import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)
otsu_val, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
global_esiкler = [50, 100, 150, 200]
blok_boyutlari = [5, 11, 21, 51]

fig, axes = plt.subplots(3, 5, figsize=(26, 14))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    f"08 — Eşikleme (Thresholding)  •  Global  |  Otsu  |  Adaptif\n"
    f"Otsu otomatik eşiği: {otsu_val:.0f}  —  İnsan gözü piksel histogramına göre en iyi ayrımı seçer",
    fontsize=17, fontweight="bold", color="white", y=0.99)

x = np.arange(256)
hist_gri = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

# ── Satır 0: Global eşikleme + Otsu ──────────────────────────────────────────
for i, t in enumerate(global_esiкler):
    _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    beyaz_oran = 100 * np.count_nonzero(th) / th.size
    axes[0, i].imshow(th, cmap="gray")
    axes[0, i].set_title(f"Global  thresh={t}", fontsize=13,
                          fontweight="bold", color="#ff8a65", pad=7)
    axes[0, i].axis("off")
    axes[0, i].set_facecolor("#0d1117")
    axes[0, i].text(0.5, -0.04, f"Beyaz: %{beyaz_oran:.1f}",
                    transform=axes[0, i].transAxes,
                    ha="center", fontsize=10, color="#b0bec5")

beyaz_oran_otsu = 100 * np.count_nonzero(otsu) / otsu.size
axes[0, 4].imshow(otsu, cmap="gray")
axes[0, 4].set_title(f"Otsu  thresh={otsu_val:.0f}\n(Otomatik)",
                      fontsize=13, fontweight="bold", color="#4fc3f7", pad=7)
axes[0, 4].axis("off")
axes[0, 4].set_facecolor("#0d1117")
axes[0, 4].text(0.5, -0.04, f"Beyaz: %{beyaz_oran_otsu:.1f}",
                transform=axes[0, 4].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

# ── Satır 1: Adaptif Gaussian ─────────────────────────────────────────────────
for i, b in enumerate(blok_boyutlari):
    adapt = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, b, 2)
    beyaz_oran = 100 * np.count_nonzero(adapt) / adapt.size
    axes[1, i].imshow(adapt, cmap="gray")
    axes[1, i].set_title(f"Adaptif Gaussian\nblok={b}×{b}",
                          fontsize=13, fontweight="bold", color="#a5d6a7", pad=7)
    axes[1, i].axis("off")
    axes[1, i].set_facecolor("#0d1117")
    axes[1, i].text(0.5, -0.04, f"Beyaz: %{beyaz_oran:.1f}",
                    transform=axes[1, i].transAxes,
                    ha="center", fontsize=10, color="#b0bec5")

# Açıklama paneli
axes[1, 4].set_facecolor("#0d1117")
axes[1, 4].text(0.5, 0.5,
                "Adaptif Eşikleme\n────────────────\n"
                "Her piksel için eşik\nlokal komşuluk bazında\nhesaplanır\n\n"
                "Blok boyutu büyüdükçe\ndaha global davranır",
                ha="center", va="center", fontsize=11, color="#a5d6a7",
                transform=axes[1, 4].transAxes)
axes[1, 4].axis("off")

# ── Satır 2: Histogram + eşik çizgileri ──────────────────────────────────────
for i, (t, renk) in enumerate(zip(global_esiкler, ["#ff5252","#ff8a65","#ffab40","#ffd740"])):
    axes[2, i].set_facecolor("#0d1117")
    axes[2, i].fill_between(x, hist_gri, alpha=0.5, color="#4fc3f7")
    axes[2, i].plot(x, hist_gri, color="#4fc3f7", linewidth=1.2)
    axes[2, i].axvline(x=t, color=renk, linewidth=2.5, linestyle="--",
                        label=f"Eşik={t}")
    axes[2, i].set_xlim(0, 255)
    axes[2, i].set_xlabel("Piksel değeri", fontsize=10, color="#b0bec5")
    if i == 0:
        axes[2, i].set_ylabel("Piksel sayısı", fontsize=10, color="#b0bec5")
    axes[2, i].legend(fontsize=10, facecolor="#1a1a2e", labelcolor="white")
    axes[2, i].tick_params(colors="#b0bec5", labelsize=9)
    axes[2, i].set_title("Histogram + Eşik", fontsize=11, color="#b0bec5")
    for sp in axes[2, i].spines.values():
        sp.set_edgecolor("#444")

axes[2, 4].set_facecolor("#0d1117")
axes[2, 4].fill_between(x, hist_gri, alpha=0.5, color="#4fc3f7")
axes[2, 4].plot(x, hist_gri, color="#4fc3f7", linewidth=1.2)
axes[2, 4].axvline(x=otsu_val, color="#ffeb3b", linewidth=2.5, linestyle="--",
                    label=f"Otsu={otsu_val:.0f}")
axes[2, 4].set_xlim(0, 255)
axes[2, 4].set_xlabel("Piksel değeri", fontsize=10, color="#b0bec5")
axes[2, 4].legend(fontsize=10, facecolor="#1a1a2e", labelcolor="white")
axes[2, 4].tick_params(colors="#b0bec5", labelsize=9)
axes[2, 4].set_title("Histogram + Otsu", fontsize=11, color="#b0bec5")
for sp in axes[2, 4].spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
