import cv2
import matplotlib.pyplot as plt
import numpy as np

img  = cv2.imread("ornek.jpg")
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

fig, axes = plt.subplots(3, 4, figsize=(22, 13))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle("04 — Renk Uzayları  •  RGB  |  Gri Ton  |  HSV  |  Lab",
             fontsize=20, fontweight="bold", color="white", y=0.99)

def goster(ax, veri, baslik, cmap=None, renk="#ffffff", ek=""):
    ax.imshow(veri, cmap=cmap)
    ax.set_title(baslik, fontsize=13, fontweight="bold", color=renk, pad=7)
    ax.axis("off")
    ax.set_facecolor("#0d1117")
    if ek:
        ax.text(0.5, -0.04, ek, transform=ax.transAxes,
                ha="center", fontsize=10, color="#b0bec5")

# Satır 1: RGB + Gri + R + G
goster(axes[0,0], rgb,        "RGB Orijinal",               renk="#4fc3f7",
       ek=f"{rgb.shape[1]}×{rgb.shape[0]} piksel")
goster(axes[0,1], gray,       "Gri Ton",        cmap="gray", renk="#b0bec5",
       ek=f"Ort: {gray.mean():.1f}   Std: {gray.std():.1f}")
goster(axes[0,2], rgb[:,:,0], "R Kanalı",       cmap="Reds",  renk="#ef5350",
       ek=f"Ort: {rgb[:,:,0].mean():.1f}")
goster(axes[0,3], rgb[:,:,1], "G Kanalı",       cmap="Greens",renk="#66bb6a",
       ek=f"Ort: {rgb[:,:,1].mean():.1f}")

# Satır 2: B + HSV
goster(axes[1,0], rgb[:,:,2], "B Kanalı",       cmap="Blues", renk="#42a5f5",
       ek=f"Ort: {rgb[:,:,2].mean():.1f}")
goster(axes[1,1], hsv[:,:,0], "H — Renk Tonu",  cmap="hsv",   renk="#fff176",
       ek="0°=Kırmızı  120°=Yeşil  240°=Mavi")
goster(axes[1,2], hsv[:,:,1], "S — Doygunluk",  cmap="gray",  renk="#fff176",
       ek="0=Gri   255=Tam doygun")
goster(axes[1,3], hsv[:,:,2], "V — Parlaklık",  cmap="gray",  renk="#fff176",
       ek="0=Siyah   255=Maksimum")

# Satır 3: Lab + histogram
goster(axes[2,0], lab[:,:,0], "L* — Aydınlık (Lab)", cmap="gray",       renk="#ce93d8",
       ek="0=Siyah  100=Beyaz  (insan algısına uygun)")
goster(axes[2,1], lab[:,:,1], "a* — Yeşil↔Kırmızı", cmap="RdYlGn_r",   renk="#ce93d8",
       ek="<128=Yeşil   >128=Kırmızı")
goster(axes[2,2], lab[:,:,2], "b* — Mavi↔Sarı",      cmap="RdYlBu_r",   renk="#ce93d8",
       ek="<128=Mavi   >128=Sarı")

axes[2,3].set_facecolor("#0d1117")
for k, (renk, etiket) in enumerate([("#ef5350","R"), ("#66bb6a","G"), ("#42a5f5","B")]):
    hist = cv2.calcHist([img], [k], None, [256], [0,256])
    axes[2,3].plot(hist, color=renk, linewidth=1.8, label=etiket, alpha=0.85)
axes[2,3].set_title("RGB Kanal Histogramları", fontsize=13,
                    fontweight="bold", color="#fff176", pad=7)
axes[2,3].set_xlabel("Piksel değeri (0–255)", fontsize=10, color="#b0bec5")
axes[2,3].legend(fontsize=11, facecolor="#1a1a2e", labelcolor="white")
axes[2,3].tick_params(colors="#b0bec5")
axes[2,3].set_xlim(0, 255)
for sp in axes[2,3].spines.values():
    sp.set_edgecolor("#444")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
