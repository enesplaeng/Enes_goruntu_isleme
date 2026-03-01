import cv2
import matplotlib.pyplot as plt
import numpy as np

gray       = cv2.cvtColor(cv2.imread("ornek.jpg"), cv2.COLOR_BGR2GRAY)
thresholds = [50, 100, 150, 200]

fig, axes = plt.subplots(3, 4, figsize=(24, 15))
fig.patch.set_facecolor("#1a1a2e")
plt.suptitle(
    "11 — Bağlı Bileşen Analizi (Connected Components)\n"
    "Binary görüntüdeki birbirine dokunan beyaz piksel grupları ayrı nesneler olarak etiketlenir",
    fontsize=17, fontweight="bold", color="white", y=0.99)

for i, t in enumerate(thresholds):
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    # connectedComponentsWithStats → alan, centroid, bbox bilgisi de verir
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # Nesne sayısı (arka plan label=0 hariç)
    nesne_sayisi = num_labels - 1

    # Alan istatistikleri (arka plan hariç)
    if nesne_sayisi > 0:
        alanlar     = stats[1:, cv2.CC_STAT_AREA]
        alan_min    = alanlar.min()
        alan_max    = alanlar.max()
        alan_ort    = alanlar.mean()
    else:
        alan_min = alan_max = alan_ort = 0

    # ── Satır 0: Binary görüntü ─────────────────────────────────────────
    beyaz_oran = 100 * np.count_nonzero(binary) / binary.size
    axes[0, i].imshow(binary, cmap="gray")
    axes[0, i].set_title(f"Binary  thresh={t}\nBeyaz: %{beyaz_oran:.1f}",
                          fontsize=13, fontweight="bold", color="#ff8a65", pad=7)
    axes[0, i].axis("off")
    axes[0, i].set_facecolor("#0d1117")

    # ── Satır 1: Renkli etiket haritası ─────────────────────────────────
    axes[1, i].imshow(labels, cmap="nipy_spectral",
                      vmin=0, vmax=max(num_labels, 1))
    axes[1, i].set_title(f"{nesne_sayisi} nesne tespit edildi",
                          fontsize=14, fontweight="bold", color="#4fc3f7", pad=7)
    axes[1, i].axis("off")
    axes[1, i].set_facecolor("#0d1117")
    axes[1, i].text(0.5, -0.04,
                    f"Alan: min={alan_min:.0f}  max={alan_max:.0f}  ort={alan_ort:.0f} px²",
                    transform=axes[1, i].transAxes,
                    ha="center", fontsize=9, color="#b0bec5")

    # ── Satır 2: En büyük 5 nesneyi kutu ile işaretle ────────────────────
    cizimlik = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if nesne_sayisi > 0:
        buyuk5 = np.argsort(alanlar)[::-1][:5]
        for rank, idx in enumerate(buyuk5):
            lbl  = idx + 1   # arka plan 0 hariç
            x    = stats[lbl, cv2.CC_STAT_LEFT]
            y    = stats[lbl, cv2.CC_STAT_TOP]
            bw   = stats[lbl, cv2.CC_STAT_WIDTH]
            bh   = stats[lbl, cv2.CC_STAT_HEIGHT]
            alan = stats[lbl, cv2.CC_STAT_AREA]
            cx, cy = int(centroids[lbl, 0]), int(centroids[lbl, 1])
            renk = [(0,255,255),(0,200,255),(0,150,255),(100,255,100),(255,200,0)][rank]
            cv2.rectangle(cizimlik, (x,y), (x+bw, y+bh), renk, 2)
            cv2.circle(cizimlik, (cx,cy), 4, renk, -1)
            cv2.putText(cizimlik, f"#{rank+1} A={alan}",
                        (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)

    axes[2, i].imshow(cv2.cvtColor(cizimlik, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title(f"En Büyük 5 Nesne (thresh={t})",
                          fontsize=12, fontweight="bold", color="#a5d6a7", pad=7)
    axes[2, i].axis("off")
    axes[2, i].set_facecolor("#0d1117")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
