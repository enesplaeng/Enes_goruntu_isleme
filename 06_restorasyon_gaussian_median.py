import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)

# Gerçekçi gösteri için yapay gürültü ekle
rng    = np.random.default_rng(42)
noise  = rng.integers(-30, 30, gray.shape, dtype=np.int16)
gurult = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

kernels = [3, 5, 9, 11]

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 20 * np.log10(255 / (np.sqrt(mse) + 1e-9))

fig, axes = plt.subplots(3, 5, figsize=(26, 13))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    "06 — Restorasyon Filtreleri  •  Gaussian vs Median\n"
    "Yapay Tuz-Biber Gürültüsü Eklendi → Her iki filtre ile temizleme karşılaştırması",
    fontsize=18, fontweight="bold", color="white", y=0.99)

# Satır 0: Orijinal ve gürültülü (sol iki sütun)
for col in range(5):
    axes[0, col].set_facecolor("#0d1117")
    axes[0, col].axis("off")

axes[0, 0].imshow(gray,   cmap="gray")
axes[0, 0].set_title("Orijinal (Temiz)", fontsize=13, fontweight="bold",
                      color="#4fc3f7", pad=8)
axes[0, 0].axis("off")

axes[0, 1].imshow(gurult, cmap="gray")
axes[0, 1].set_title("Gürültülü Görüntü\n(±30 rassal gürültü eklendi)",
                      fontsize=13, fontweight="bold", color="#ef5350", pad=8)
axes[0, 1].axis("off")

# Boş hücrelere açıklama yaz
for col in [2, 3, 4]:
    axes[0, col].text(0.5, 0.5,
                      ["Küçük çekirdek\ndaha az bulanıklık",
                       "Orta çekirdek\ndenge noktası",
                       "Büyük çekirdek\ngürültü ↓  detay ↓"][col-2],
                      ha="center", va="center", fontsize=11, color="#b0bec5",
                      transform=axes[0, col].transAxes)

# Satır 1: Gaussian
for i, k in enumerate(kernels):
    sonuc = cv2.GaussianBlur(gurult, (k, k), 0)
    p = psnr(gray, sonuc)
    axes[1, i].imshow(sonuc, cmap="gray")
    axes[1, i].set_title(f"Gaussian  {k}×{k}", fontsize=13,
                          fontweight="bold", color="#a5d6a7", pad=7)
    axes[1, i].axis("off")
    axes[1, i].set_facecolor("#0d1117")
    axes[1, i].text(0.5, -0.04, f"PSNR: {p:.1f} dB",
                    transform=axes[1, i].transAxes,
                    ha="center", fontsize=11, color="#b0bec5")

axes[1, 4].set_facecolor("#0d1117")
axes[1, 4].text(0.5, 0.5,
                "Gaussian Blur\n────────────\n"
                "Kenarları yumuşatır\nHer piksel komşularının\nağırlıklı ortalaması\n\n"
                "σ büyüdükçe\ndaha fazla bulanıklık",
                ha="center", va="center", fontsize=11, color="#a5d6a7",
                transform=axes[1, 4].transAxes)
axes[1, 4].axis("off")

# Satır 2: Median
for i, k in enumerate(kernels):
    sonuc = cv2.medianBlur(gurult, k)
    p = psnr(gray, sonuc)
    axes[2, i].imshow(sonuc, cmap="gray")
    axes[2, i].set_title(f"Median  {k}×{k}", fontsize=13,
                          fontweight="bold", color="#fff176", pad=7)
    axes[2, i].axis("off")
    axes[2, i].set_facecolor("#0d1117")
    axes[2, i].text(0.5, -0.04, f"PSNR: {p:.1f} dB",
                    transform=axes[2, i].transAxes,
                    ha="center", fontsize=11, color="#b0bec5")

axes[2, 4].set_facecolor("#0d1117")
axes[2, 4].text(0.5, 0.5,
                "Median Filtre\n────────────\n"
                "Tuz-biber gürültüsüne\nçok daha etkili!\n\n"
                "Her piksel komşularının\nmedyan değerini alır\n\n"
                "Kenar koruması daha iyi",
                ha="center", va="center", fontsize=11, color="#fff176",
                transform=axes[2, 4].transAxes)
axes[2, 4].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
