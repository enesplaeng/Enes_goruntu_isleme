import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)
otsu_val, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernels = [3, 7, 11]

ISLEMLER = [
    ("Dilation\n(Genişletme)",  cv2.MORPH_DILATE,  "#ef5350",
     "Beyaz bölgeler büyür\nSiyah delikler kapanır"),
    ("Erosion\n(Aşındırma)",    cv2.MORPH_ERODE,   "#42a5f5",
     "Beyaz bölgeler küçülür\nGürültü nesneleri yok olur"),
    ("Opening\n(Açma)",         cv2.MORPH_OPEN,    "#a5d6a7",
     "Erosion → Dilation\nKüçük beyaz noktaları siler"),
    ("Closing\n(Kapama)",       cv2.MORPH_CLOSE,   "#fff176",
     "Dilation → Erosion\nKüçük siyah delikleri doldurur"),
]

fig, axes = plt.subplots(len(kernels) + 1, 5, figsize=(26, 16))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    f"07 — Morfolojik İşlemler  •  İkili (Binary) Görüntü Üzerinde\n"
    f"Otsu Eşiği: {otsu_val:.0f}  |  Yapılandırma Elemanı: Dikdörtgen (MORPH_RECT)",
    fontsize=18, fontweight="bold", color="white", y=0.99)

# Başlık satırı (0. satır)
for j in range(5):
    axes[0, j].set_facecolor("#0d1117")
    axes[0, j].axis("off")

axes[0, 0].imshow(binary, cmap="gray")
axes[0, 0].set_title(f"Orijinal Binary\nOtsu={otsu_val:.0f}",
                      fontsize=13, fontweight="bold", color="#4fc3f7", pad=7)
axes[0, 0].axis("off")

for j, (islem, _, renk, aciklama) in enumerate(ISLEMLER):
    axes[0, j+1].text(0.5, 0.5, f"{islem}\n\n{aciklama}",
                      ha="center", va="center", fontsize=11,
                      color=renk, transform=axes[0, j+1].transAxes,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
                                edgecolor=renk, alpha=0.7))

# İşlem satırları
for i, k in enumerate(kernels):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    beyaz  = np.count_nonzero(binary)

    axes[i+1, 0].imshow(binary, cmap="gray")
    axes[i+1, 0].set_title(f"Çekirdek {k}×{k}", fontsize=12,
                             fontweight="bold", color="#b0bec5", pad=6)
    axes[i+1, 0].axis("off")
    axes[i+1, 0].set_facecolor("#0d1117")

    for j, (islem, morph_tip, renk, _) in enumerate(ISLEMLER):
        if morph_tip == cv2.MORPH_DILATE:
            sonuc = cv2.dilate(binary, kernel)
        elif morph_tip == cv2.MORPH_ERODE:
            sonuc = cv2.erode(binary, kernel)
        else:
            sonuc = cv2.morphologyEx(binary, morph_tip, kernel)

        fark = int(np.count_nonzero(sonuc)) - beyaz
        fark_str = f"+{fark}" if fark >= 0 else str(fark)

        axes[i+1, j+1].imshow(sonuc, cmap="gray")
        baslik_kisa = islem.split("\n")[0]
        axes[i+1, j+1].set_title(f"{baslik_kisa}  {k}×{k}",
                                   fontsize=12, fontweight="bold", color=renk, pad=6)
        axes[i+1, j+1].axis("off")
        axes[i+1, j+1].set_facecolor("#0d1117")
        axes[i+1, j+1].text(0.5, -0.04, f"Beyaz piksel değişimi: {fark_str}",
                             transform=axes[i+1, j+1].transAxes,
                             ha="center", fontsize=10, color="#b0bec5")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
