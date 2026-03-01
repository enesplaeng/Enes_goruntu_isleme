import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)

canny_params = [(50, 100), (100, 200), (150, 300)]
sobelx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel    = cv2.magnitude(sobelx, sobely)
laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    "09 — Kenar Algılama (Edge Detection)\n"
    "Canny (çift eşikli)  |  Sobel (gradyan türevi)  |  Laplacian (2. türev)",
    fontsize=18, fontweight="bold", color="white", y=0.99)

# ── Üst satır: 3× Canny + Sobel X ────────────────────────────────────────────
for i, (low, high) in enumerate(canny_params):
    edges = cv2.Canny(gray, low, high)
    kenar_sayisi = np.count_nonzero(edges)
    axes[0, i].imshow(edges, cmap="gray")
    axes[0, i].set_title(f"Canny  [{low}–{high}]",
                          fontsize=14, fontweight="bold", color="#4fc3f7", pad=8)
    axes[0, i].axis("off")
    axes[0, i].set_facecolor("#0d1117")
    axes[0, i].text(0.5, -0.04,
                    f"Kenar piksel: {kenar_sayisi:,}  (%{100*kenar_sayisi/edges.size:.1f})",
                    transform=axes[0, i].transAxes,
                    ha="center", fontsize=10, color="#b0bec5")

sx_norm = np.clip(np.abs(sobelx) / np.abs(sobelx).max() * 255, 0, 255).astype(np.uint8)
axes[0, 3].imshow(sx_norm, cmap="gray")
axes[0, 3].set_title("Sobel X  (Yatay Kenarlar)",
                      fontsize=14, fontweight="bold", color="#ff8a65", pad=8)
axes[0, 3].axis("off")
axes[0, 3].set_facecolor("#0d1117")
axes[0, 3].text(0.5, -0.04, "∂I/∂x  —  Dikey çizgileri yakalar",
                transform=axes[0, 3].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

# ── Alt satır: Sobel Y, Sobel Combined, Laplacian, Açıklama ──────────────────
sy_norm = np.clip(np.abs(sobely) / np.abs(sobely).max() * 255, 0, 255).astype(np.uint8)
axes[1, 0].imshow(sy_norm, cmap="gray")
axes[1, 0].set_title("Sobel Y  (Dikey Kenarlar)",
                      fontsize=14, fontweight="bold", color="#ff8a65", pad=8)
axes[1, 0].axis("off")
axes[1, 0].set_facecolor("#0d1117")
axes[1, 0].text(0.5, -0.04, "∂I/∂y  —  Yatay çizgileri yakalar",
                transform=axes[1, 0].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

sob_norm = np.clip(sobel / sobel.max() * 255, 0, 255).astype(np.uint8)
axes[1, 1].imshow(sob_norm, cmap="gray")
axes[1, 1].set_title("Sobel Birleşik  √(Sx²+Sy²)",
                      fontsize=14, fontweight="bold", color="#ffcc80", pad=8)
axes[1, 1].axis("off")
axes[1, 1].set_facecolor("#0d1117")
axes[1, 1].text(0.5, -0.04, "Gradyan büyüklüğü — tüm yönler",
                transform=axes[1, 1].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

lap_norm = np.clip(laplacian / laplacian.max() * 255, 0, 255).astype(np.uint8)
axes[1, 2].imshow(lap_norm, cmap="gray")
axes[1, 2].set_title("Laplacian  ∇²I",
                      fontsize=14, fontweight="bold", color="#ce93d8", pad=8)
axes[1, 2].axis("off")
axes[1, 2].set_facecolor("#0d1117")
axes[1, 2].text(0.5, -0.04, "2. türev — yön bağımsız, gürültüye duyarlı",
                transform=axes[1, 2].transAxes,
                ha="center", fontsize=10, color="#b0bec5")

# Karşılaştırma özet paneli
axes[1, 3].set_facecolor("#0d1117")
axes[1, 3].axis("off")
ozet = (
    "Algoritma Karşılaştırması\n"
    "─────────────────────\n\n"
    "Canny\n"
    "  ✔ En iyi sonuç\n"
    "  ✔ Çift eşik ile ince kenar\n"
    "  ✔ Gürültüye dayanıklı\n\n"
    "Sobel\n"
    "  ✔ Yön bilgisi verir\n"
    "  ✔ Hızlı hesap\n"
    "  ✘ Gürültüye orta duyarlı\n\n"
    "Laplacian\n"
    "  ✔ Yön bağımsız\n"
    "  ✘ Gürültüye çok duyarlı"
)
axes[1, 3].text(0.5, 0.5, ozet, ha="center", va="center",
                fontsize=10.5, color="#e0e0e0",
                transform=axes[1, 3].transAxes,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                          edgecolor="#555", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
