import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# PASCAL VOC 21 sınıf isimleri (İngilizce + Türkçe)
SINIFLAR = [
    ("background",   "Arka Plan",    "#000000"),
    ("aeroplane",    "Uçak",         "#800000"),
    ("bicycle",      "Bisiklet",     "#008000"),
    ("bird",         "Kuş",          "#800080"),
    ("boat",         "Tekne",        "#808000"),
    ("bottle",       "Şişe",         "#000080"),
    ("bus",          "Otobüs",       "#008080"),
    ("car",          "Araba",        "#808080"),
    ("cat",          "Kedi",         "#c00000"),
    ("chair",        "Sandalye",     "#00c000"),
    ("cow",          "İnek",         "#c0c000"),
    ("diningtable",  "Masa",         "#0000c0"),
    ("dog",          "Köpek",        "#c000c0"),
    ("horse",        "At",           "#00c0c0"),
    ("motorbike",    "Motorsiklet",  "#c0c0c0"),
    ("person",       "İnsan",        "#400000"),
    ("pottedplant",  "Saksı Bitki",  "#004000"),
    ("sheep",        "Koyun",        "#400040"),
    ("sofa",         "Kanepe",       "#404000"),
    ("train",        "Tren",         "#000040"),
    ("tvmonitor",    "TV/Monitör",   "#004040"),
]

# ── Model yükle ───────────────────────────────────────────────────────────────
print("DeepLabV3-ResNet50 modeli yükleniyor...")
weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
model   = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
model.eval()
print("Model hazır.")

# ── Görüntü hazırla ───────────────────────────────────────────────────────────
img       = Image.open("ornek.jpg").convert("RGB")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]),
])
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(inp)["out"][0]

mask      = out.argmax(0).cpu().numpy()
bulunan   = [i for i in np.unique(mask) if i > 0]   # arka plan hariç

# ── Renkli maske oluştur ──────────────────────────────────────────────────────
mask_renkli = np.zeros((*mask.shape, 3), dtype=np.uint8)
for i, (_, _, hex_renk) in enumerate(SINIFLAR):
    r = int(hex_renk[1:3], 16)
    g = int(hex_renk[3:5], 16)
    b = int(hex_renk[5:7], 16)
    mask_renkli[mask == i] = [r, g, b]

img_arr = np.array(img)

# ── Figür ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 9))
fig.patch.set_facecolor("#1a1a2e")

sinif_ozeti = " | ".join([SINIFLAR[i][1] for i in bulunan]) if bulunan else "—"
plt.suptitle(
    f"12 — Derin Öğrenme ile Semantik Segmentasyon  •  DeepLabV3-ResNet50\n"
    f"PASCAL VOC 21 sınıf  |  Tespit edilen: {sinif_ozeti}",
    fontsize=17, fontweight="bold", color="white", y=0.99)

# Sol: Orijinal
ax1 = fig.add_subplot(1, 4, 1)
ax1.imshow(img_arr)
ax1.set_title("Orijinal Görüntü",
              fontsize=13, fontweight="bold", color="#4fc3f7", pad=8)
ax1.axis("off")
ax1.set_facecolor("#0d1117")
ax1.text(0.5, -0.03, f"{img.width}×{img.height} piksel",
         transform=ax1.transAxes, ha="center", fontsize=10, color="#b0bec5")

# Orta-sol: Segmentasyon maskesi
ax2 = fig.add_subplot(1, 4, 2)
ax2.imshow(mask_renkli)
ax2.set_title("Segmentasyon Maskesi\n(her renk = bir sınıf)",
              fontsize=13, fontweight="bold", color="#a5d6a7", pad=8)
ax2.axis("off")
ax2.set_facecolor("#0d1117")

# Orta-sağ: Overlay
ax3 = fig.add_subplot(1, 4, 3)
ax3.imshow(img_arr)
ax3.imshow(mask_renkli, alpha=0.45)
ax3.set_title("Overlay (Şeffaf Maske)",
              fontsize=13, fontweight="bold", color="#fff176", pad=8)
ax3.axis("off")
ax3.set_facecolor("#0d1117")

# Sağ: Sınıf legendı
ax4 = fig.add_subplot(1, 4, 4)
ax4.set_facecolor("#0d1117")
ax4.axis("off")
ax4.set_title("Tespit Edilen Sınıflar",
              fontsize=13, fontweight="bold", color="#ce93d8", pad=8)

yama_listesi = []
for sinif_idx in bulunan:
    eng, tur, hex_r = SINIFLAR[sinif_idx]
    r = int(hex_r[1:3], 16) / 255
    g = int(hex_r[3:5], 16) / 255
    b = int(hex_r[5:7], 16) / 255
    piksel_sayisi = int(np.sum(mask == sinif_idx))
    oran          = 100 * piksel_sayisi / mask.size
    yama = mpatches.Patch(color=(r, g, b),
                           label=f"{tur} ({eng})  %{oran:.1f}")
    yama_listesi.append(yama)

if yama_listesi:
    ax4.legend(handles=yama_listesi, loc="center", fontsize=11,
               facecolor="#0d1117", labelcolor="white",
               edgecolor="#555", framealpha=0.8)
else:
    ax4.text(0.5, 0.5, "Nesne bulunamadı\n(sadece arka plan)",
             ha="center", va="center", fontsize=12, color="#b0bec5",
             transform=ax4.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
