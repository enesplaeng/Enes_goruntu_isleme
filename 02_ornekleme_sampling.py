import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("ornek.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
factors = [2, 4, 8]

def psnr(orig, proc):
    mse = np.mean((orig.astype(np.float32) - proc.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle("02 — Örnekleme (Sampling)  •  Çözünürlük Azaltınca Ne Olur?",
             fontsize=19, fontweight="bold", color="white", y=0.98)

# Üst satır: görüntüler
axes[0, 0].imshow(rgb)
axes[0, 0].set_title(f"Orijinal\n{w}×{h} piksel", fontsize=13, fontweight="bold",
                     color="#4fc3f7", pad=8)
axes[0, 0].axis("off")

for i, f in enumerate(factors):
    small = cv2.resize(rgb, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    big   = cv2.resize(small, (w, h),         interpolation=cv2.INTER_NEAREST)
    p = psnr(rgb, big)
    axes[0, i+1].imshow(big)
    axes[0, i+1].set_title(f"1/{f} Örnekleme\n{w//f}×{h//f} → {w}×{h}",
                            fontsize=13, fontweight="bold", color="#ff8a65", pad=8)
    axes[0, i+1].axis("off")
    axes[0, i+1].text(0.5, -0.03, f"PSNR: {p:.1f} dB",
                      transform=axes[0, i+1].transAxes,
                      ha="center", fontsize=11, color="#ffcc80")

# Alt satır: fark görüntüsü (hata haritası)
axes[1, 0].set_facecolor("#0d1117")
axes[1, 0].text(0.5, 0.5,
                "Hata Haritası\n(Orijinal − İşlenmiş)\n\nSıcak renk = büyük hata",
                ha="center", va="center", fontsize=12, color="#b0bec5",
                transform=axes[1, 0].transAxes)
axes[1, 0].axis("off")

cmap_err = plt.get_cmap("hot")
for i, f in enumerate(factors):
    small = cv2.resize(rgb, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    big   = cv2.resize(small, (w, h),         interpolation=cv2.INTER_NEAREST)
    diff  = np.abs(rgb.astype(np.float32) - big.astype(np.float32)).mean(axis=2)
    im = axes[1, i+1].imshow(diff, cmap="hot", vmin=0, vmax=80)
    axes[1, i+1].set_title(f"Hata — 1/{f}\nMSE: {diff.mean():.1f}",
                            fontsize=12, color="#ef9a9a", pad=6)
    axes[1, i+1].axis("off")
    plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.set_facecolor("#0d1117")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
