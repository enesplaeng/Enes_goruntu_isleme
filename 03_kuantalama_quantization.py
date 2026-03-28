import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)
bits = [8, 3, 2, 1]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle("03 — Kuantalama (Quantization)  •  Bit Derinliği Azalınca Renk Kaybı",
             fontsize=19, fontweight="bold", color="white", y=0.98)

renk_baslik = ["#4fc3f7", "#a5d6a7", "#ffcc80", "#ef9a9a"]

for i, b in enumerate(bits):
    levels = 2 ** b
    step   = 256 // levels
    quant  = (gray // step) * step
    mse    = np.mean((gray.astype(np.float32) - quant.astype(np.float32)) ** 2)
    psnr   = 20 * np.log10(255 / (np.sqrt(mse) + 1e-9))

    # Üst satır: görüntü
    axes[0, i].imshow(quant, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(
        f"{b}-bit  →  {levels} renk seviyesi\nAdım büyüklüğü: {step}",
        fontsize=13, fontweight="bold", color=renk_baslik[i], pad=8)
    axes[0, i].axis("off")
    axes[0, i].text(0.5, -0.03, f"PSNR: {psnr:.1f} dB  |  MSE: {mse:.1f}",
                    transform=axes[0, i].transAxes,
                    ha="center", fontsize=11, color="#b0bec5")

    # Alt satır: histogram
    axes[1, i].set_facecolor("#0d1117")
    hist_orig  = cv2.calcHist([gray],  [0], None, [256], [0, 256]).flatten()
    hist_quant = cv2.calcHist([quant], [0], None, [256], [0, 256]).flatten()
    x = np.arange(256)
    axes[1, i].fill_between(x, hist_orig,  alpha=0.35, color="#4fc3f7", label="Orijinal")
    axes[1, i].fill_between(x, hist_quant, alpha=0.70, color=renk_baslik[i], label=f"{b}-bit")
    axes[1, i].set_title(f"Histogram karşılaştırması", fontsize=11, color="#b0bec5")
    axes[1, i].set_xlabel("Piksel değeri", fontsize=10, color="#b0bec5")
    axes[1, i].legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    axes[1, i].tick_params(colors="#b0bec5", labelsize=9)
    for spine in axes[1, i].spines.values():
        spine.set_edgecolor("#444")

for ax in axes[0]:
    ax.set_facecolor("#0d1117")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
