import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

img = cv2.imread("ornek.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Z   = img.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
k_values  = [2, 4, 8, 16]

fig, axes = plt.subplots(2, 5, figsize=(26, 10))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    "10 — K-Means Renk Segmentasyonu\n"
    "Her piksel en yakın renk merkezine atanır — K arttıkça daha fazla detay korunur",
    fontsize=18, fontweight="bold", color="white", y=0.99)

# Orijinal
axes[0, 0].imshow(rgb)
axes[0, 0].set_title(f"Orijinal\n{rgb.shape[1]}×{rgb.shape[0]}",
                      fontsize=13, fontweight="bold", color="#4fc3f7", pad=8)
axes[0, 0].axis("off")
axes[0, 0].set_facecolor("#0d1117")
axes[0, 0].text(0.5, -0.04, f"Teorik renk: {256**3:,}",
                transform=axes[0, 0].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

# Palet paneli (orijinal için de boş göster)
axes[1, 0].set_facecolor("#0d1117")
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5,
                "K-Means Algoritması\n────────────────\n"
                "1. K merkez rastgele seç\n"
                "2. Her pikseli en yakın\n   merkeze ata\n"
                "3. Merkezleri güncelle\n"
                "4. Yakınsayana dek tekrar",
                ha="center", va="center", fontsize=10, color="#b0bec5",
                transform=axes[1, 0].transAxes)

renkler_baslik = ["#ef9a9a", "#a5d6a7", "#fff176", "#ce93d8"]

for i, K in enumerate(k_values):
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10,
                                     cv2.KMEANS_RANDOM_CENTERS)
    centers_u8  = np.uint8(centers)
    segmented   = centers_u8[labels.flatten()].reshape(img.shape)
    seg_rgb     = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

    # Üst satır: segmentasyon sonucu
    axes[0, i+1].imshow(seg_rgb)
    axes[0, i+1].set_title(f"K = {K}  ({K} renk)",
                             fontsize=13, fontweight="bold",
                             color=renkler_baslik[i], pad=8)
    axes[0, i+1].axis("off")
    axes[0, i+1].set_facecolor("#0d1117")
    axes[0, i+1].text(0.5, -0.04,
                      f"Veri sıkıştırma: {100*(1 - K*3/(256**3*3)):.4f}%",
                      transform=axes[0, i+1].transAxes,
                      ha="center", fontsize=9, color="#b0bec5")

    # Alt satır: renk paleti
    axes[1, i+1].set_facecolor("#0d1117")
    axes[1, i+1].axis("off")
    palette_rgb = cv2.cvtColor(
        centers_u8.reshape(1, K, 3), cv2.COLOR_BGR2RGB
    ).reshape(K, 3)

    bar_h = 1.0 / K
    for j, renk in enumerate(palette_rgb):
        rect = mpatches.FancyBboxPatch(
            (0.05, j * bar_h + 0.01), 0.55, bar_h - 0.02,
            boxstyle="round,pad=0.01",
            facecolor=renk / 255.0, edgecolor="none"
        )
        axes[1, i+1].add_patch(rect)
        axes[1, i+1].text(0.65, j * bar_h + bar_h / 2,
                          f"R{renk[0]} G{renk[1]} B{renk[2]}",
                          va="center", fontsize=max(6, 9 - K // 4),
                          color="#e0e0e0",
                          transform=axes[1, i+1].transAxes)
    axes[1, i+1].set_title(f"Renk Paleti (K={K})",
                            fontsize=11, color=renkler_baslik[i], pad=4)
    axes[1, i+1].set_xlim(0, 1)
    axes[1, i+1].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
