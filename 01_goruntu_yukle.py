import cv2
import matplotlib.pyplot as plt

# ── Görüntüyü oku ─────────────────────────────────────────────────────────────
img  = cv2.imread("ytu.jpg")
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]

# ── Figür ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor("#1a1a2e")

plt.suptitle("01 — Görüntü Yükleme ve Temel Bilgiler",
             fontsize=20, fontweight="bold", color="white", y=0.97)

# Sol: Renkli
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(rgb)
ax1.set_title("Renkli resim (RGB)", fontsize=15, fontweight="bold", color="#4fc3f7", pad=10)
ax1.axis("off")
ax1.text(0.5, -0.04,
         f"Boyut: {w} × {h} piksel   |   Kanal: 3 (R, G, B)   |   Dtype: {img.dtype}",
         transform=ax1.transAxes, ha="center", fontsize=11, color="#b0bec5")

# Orta: Gri ton
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(gray, cmap="gray")
ax2.set_title("Gri Tonlu resim (Grayscale)", fontsize=15, fontweight="bold", color="#a5d6a7", pad=10)
ax2.axis("off")
ax2.text(0.5, -0.04,
         f"Min: {gray.min()}   Max: {gray.max()}   Ortalama: {gray.mean():.1f}",
         transform=ax2.transAxes, ha="center", fontsize=11, color="#b0bec5")

# Sağ: Histogram
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor("#0d1117")
renkler = [("#ef5350", "R Kanalı"), ("#66bb6a", "G Kanalı"), ("#42a5f5", "B Kanalı")]
for k, (renk, etiket) in enumerate(renkler):
    hist = cv2.calcHist([img], [k], None, [256], [0, 256])
    ax3.plot(hist, color=renk, linewidth=1.5, label=etiket, alpha=0.85)
ax3.set_title("Piksel Yoğunluk Histogramı", fontsize=15, fontweight="bold",
              color="#fff176", pad=10)
ax3.set_xlabel("Piksel Değeri (0–255)", fontsize=11, color="#b0bec5")
ax3.set_ylabel("Frekans", fontsize=11, color="#b0bec5")
ax3.legend(fontsize=11, facecolor="#1a1a2e", labelcolor="white")
ax3.tick_params(colors="#b0bec5")
ax3.set_xlim(0, 255)
for spine in ax3.spines.values():
    spine.set_edgecolor("#444")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
