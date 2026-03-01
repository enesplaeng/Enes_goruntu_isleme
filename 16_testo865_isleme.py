"""
16 - Testo 865 Termal Kamera Görüntü İşleme
============================================
Testo 865 termal kamerasından gelen veriyi iki modda işler:

  MOD 1 — DOSYA (--mod dosya):
    Testo 865'in USB ile aktardığı .bmt / .jpg / .xlsx / .csv dosyalarını
    bir klasörü izleyerek otomatik işler.  Yeni dosya gelince analiz eder.

    Excel/CSV formatı (Testo IRSoft dışa aktarımı):
      - Her satır bir piksel satırı, her sütun bir piksel sütunu
      - Hücre değerleri °C cinsinden gerçek sıcaklık
      - Örnek: 120 satır × 160 sütun = Testo 865 tam çözünürlük

  MOD 2 — CANLI (--mod canli):
    Kamera Linux'ta UVC (/dev/video*) olarak görünüyorsa gerçek
    zamanlı işler; görünmüyorsa webcam üzerinde termal simülasyon yapar.

İşaretleme Özellikleri:
  • Sıcak nokta  (max °C) — kırmızı daire + etiket
  • Soğuk nokta  (min °C) — mavi daire + etiket
  • Eşik alarmı  — belirli °C üstündeki tüm bölge renkli maske
  • Sıcaklık profili çizgisi — yatay tarama çizgisi + grafik
  • Renk çubuğu + histogram
  • Piksel bazlı sıcaklık okuma (fareyle üzerine gel)

Çalıştırma:
  python3 16_testo865_isleme.py --mod dosya  --klasor /path/to/testo_output
  python3 16_testo865_isleme.py --mod canli

Kontroller:
  [Mouse]   : Üzerine gelince piksel sıcaklığı
  [t]       : Eşik değerini +1°C artır
  [T]       : Eşik değerini -1°C azalt
  [e]       : Eşik alarm maskesini aç/kapat
  [p]       : Profil çizgisini aç/kapat
  [1-7]     : Termal renk paleti
  [s]       : Kaydet
  [q]       : Çıkış
"""

import cv2
import numpy as np
import time
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

# ─── Testo 865 Teknik Özellikler (Datasheet) ─────────────────────────────────
#
#  Çözünürlük   : 160×120 piksel (native) | 320×240 (SuperResolution)
#  Yenileme     : 9 Hz  →  ~111 ms/kare
#  Ölçüm aralığı: -20 … +280 °C
#  Doğruluk     : ±2 °C veya ±2% (büyük olan)
#  Duyarlılık   : <0.1 °C (100 mK)
#  Spektral band: 7.5 … 14 µm
#  Dosya formatı: .bmt | .jpg | .bmp | .png | .xls | .csv
#  USB canlı yay: USB 2.0 Micro-B (UVC)
#  Testo paleti : Demir | Gökkuşağı | Soğuk-Sıcak | Gri
#
# ─────────────────────────────────────────────────────────────────────────────

T_MIN_GERCEK = -20.0    # °C — Testo 865 minimum ölçüm sınırı
T_MAX_GERCEK = 280.0    # °C — Testo 865 maksimum ölçüm sınırı
DOGRULUK_C   =   2.0    # °C — ±2°C veya ±%2 (datasheet)

TESTO_COZUNURLUK    = (160, 120)   # native çözünürlük
TESTO_SUPER_RES     = (320, 240)   # SuperResolution modu
TESTO_FPS           = 9            # 9 Hz yenileme (datasheet)
TESTO_BEKLEME_MS    = 111          # 1000 / 9 Hz ≈ 111 ms

GOSTERIM_BOYUTU = (640, 480)

# Testo 865'in desteklediği 4 resmi palet (datasheet: demir, gökkuşağı, soğuk-sıcak, gri)
# OpenCV'de tam karşılıklar:
PALETLER = [
    ("Demir      (Iron)",     cv2.COLORMAP_HOT),      # Testo "demir" paleti
    ("Gokusakagi (Rainbow)",  cv2.COLORMAP_RAINBOW),  # Testo "gökkuşağı"
    ("Soguk-Sicak",           cv2.COLORMAP_JET),      # Testo "soğuk-sıcak"
    ("Gri        (Gray)",     cv2.COLORMAP_BONE),     # Testo "gri"
]

# BMT dosyasında JPEG SOI işaretçisi
JPEG_SOI = b'\xff\xd8\xff'

PENCERE_ADI = "Testo 865 — Termal Goruntu Isleme  (160x120 | 9Hz | -20..+280C)"

# ─── Sıcaklık Dönüşümleri ────────────────────────────────────────────────────

def piksel_to_sicaklik(deger: float,
                        t_min: float = T_MIN_GERCEK,
                        t_max: float = T_MAX_GERCEK) -> float:
    """0-255 piksel değerini °C'ye çevirir (doğrusal interpolasyon)."""
    return t_min + (deger / 255.0) * (t_max - t_min)


def sicaklik_to_piksel(sicaklik: float,
                        t_min: float = T_MIN_GERCEK,
                        t_max: float = T_MAX_GERCEK) -> int:
    """°C değerini 0-255 piksel değerine çevirir."""
    return int(np.clip((sicaklik - t_min) / (t_max - t_min) * 255, 0, 255))


# ─── Testo BMT Okuyucu ───────────────────────────────────────────────────────

class TestoBMTOkuyucu:
    """
    Testo BMT dosyasından görüntü ve (varsa) ham sıcaklık verisi çıkarır.

    BMT Format Özeti:
      - Dosya binary, Testo'ya özel yapı
      - İçinde gömülü bir JPEG (görsel/termal renk) bulunur
      - Ham ısı verisi (int16, little-endian) belirli bir ofsette saklanır
      - Ofset kamera modeline göre değişebilir; burada Testo 865 için
        yaygın değer kullanılmıştır (0xE4 bayt başlık)
    """

    # Testo 865 — ham veri ofseti ve boyutu (topluluk tarafından tespit edilmiş)
    HAM_VERİ_OFSET   = 0x00E4        # bayt
    PIKSEL_SAYISI     = 160 * 120     # 19200 piksel
    SICAKLIK_KATSAYI  = 0.01          # ham_deger * 0.01 = °C (Kelvin'den)
    SICAKLIK_OFFSET_K = 273.15        # Kelvin ofset

    @staticmethod
    def bmt_oku(dosya_yolu: str) -> tuple[np.ndarray | None,
                                           np.ndarray | None,
                                           dict]:
        """
        BMT dosyasını okur.

        Returns:
            gorsel   : BGR görüntü (gömülü JPEG)
            ham_C    : float32 sıcaklık matrisi °C cinsinden (yoksa None)
            meta     : {'t_min', 't_max', 't_ort', 'kaynak'} sözlüğü
        """
        try:
            with open(dosya_yolu, 'rb') as f:
                veri = f.read()
        except OSError as e:
            print(f"Dosya okunamadi: {e}")
            return None, None, {}

        # ── Ham sıcaklık verisi ──────────────────────────────────────────────
        ham_C = None
        ofset = TestoBMTOkuyucu.HAM_VERİ_OFSET
        n     = TestoBMTOkuyucu.PIKSEL_SAYISI
        if len(veri) >= ofset + n * 2:
            ham = np.frombuffer(veri[ofset: ofset + n * 2],
                                dtype='<i2').astype(np.float32)
            sicaklik = ham * TestoBMTOkuyucu.SICAKLIK_KATSAYI \
                       - TestoBMTOkuyucu.SICAKLIK_OFFSET_K
            # Makul sıcaklık aralığında mı kontrol et (-40 … 400°C)
            if -40 < sicaklik.mean() < 400:
                ham_C = sicaklik.reshape(120, 160)

        # ── Gömülü JPEG ─────────────────────────────────────────────────────
        gorsel = None
        idx = veri.find(JPEG_SOI)
        if idx != -1:
            jpeg_veri = np.frombuffer(veri[idx:], dtype=np.uint8)
            gorsel = cv2.imdecode(jpeg_veri, cv2.IMREAD_COLOR)

        # ── Meta ─────────────────────────────────────────────────────────────
        meta = {}
        if ham_C is not None:
            meta = {
                't_min': float(ham_C.min()),
                't_max': float(ham_C.max()),
                't_ort': float(ham_C.mean()),
                'kaynak': 'ham_veri',
            }
        elif gorsel is not None:
            # Görsel yoksa yoksa gri tondan tahmin et
            gri = cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY).astype(np.float32)
            meta = {
                't_min': piksel_to_sicaklik(float(gri.min())),
                't_max': piksel_to_sicaklik(float(gri.max())),
                't_ort': piksel_to_sicaklik(float(gri.mean())),
                'kaynak': 'gorsel_tahmini',
            }

        return gorsel, ham_C, meta

    @staticmethod
    def jpg_oku(dosya_yolu: str) -> tuple[np.ndarray | None,
                                           np.ndarray | None,
                                           dict]:
        """
        Testo'nun dışa aktardığı radiometric JPEG / standart JPEG'i okur.
        Ham veri yoksa gri tondan sıcaklık tahmini yapar.
        """
        gorsel = cv2.imread(dosya_yolu)
        if gorsel is None:
            return None, None, {}

        gri = cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ham_C = None  # JPEG'de ham sıcaklık yok (radiometric değilse)

        meta = {
            't_min': piksel_to_sicaklik(float(gri.min())),
            't_max': piksel_to_sicaklik(float(gri.max())),
            't_ort': piksel_to_sicaklik(float(gri.mean())),
            'kaynak': 'gorsel_tahmini',
        }
        return gorsel, ham_C, meta

    @staticmethod
    def excel_oku(dosya_yolu: str) -> tuple[np.ndarray | None,
                                             np.ndarray | None,
                                             dict]:
        """
        Testo IRSoft'un dışa aktardığı Excel (.xlsx) veya CSV (.csv) dosyasını okur.

        Beklenen format (Testo IRSoft standart çıktısı):
          - İlk satırlar metadata (başlık, tarih, emissivity vb.) — atlanır
          - Sayısal değerler içeren blok: satır=Y piksel, sütun=X piksel
          - Değerler °C cinsinden float

        Testo IRSoft bazen üst kısma bilgi satırları ekler; fonksiyon
        tamamı sayısal olan ilk bloğu otomatik bulur.
        """
        try:
            ext = Path(dosya_yolu).suffix.lower()
            if ext in ('.xlsx', '.xls'):
                # Tüm sayfayı oku, header yok (metadata satırları var olabilir)
                df_raw = pd.read_excel(dosya_yolu, header=None,
                                       engine='openpyxl')
            else:  # .csv
                df_raw = pd.read_csv(dosya_yolu, header=None,
                                     sep=None, engine='python')
        except Exception as e:
            print(f"Excel/CSV okunamadi: {e}")
            return None, None, {}

        # ── Sayısal bloğu bul ────────────────────────────────────────────────
        # Her satırı sayısala çevirmeye çalış; başarısız olanları atla
        satirlar = []
        for _, row in df_raw.iterrows():
            sayisal = pd.to_numeric(row, errors='coerce')
            if sayisal.notna().sum() >= 5:   # en az 5 geçerli sayı varsa veri satırı
                satirlar.append(sayisal.to_numpy(dtype=np.float32,
                                                  na_value=np.nan))

        if not satirlar:
            print("Excel'de sayisal sicaklik verisi bulunamadi.")
            return None, None, {}

        # Sütun sayısını standardize et (en uzun satıra göre)
        max_sutun = max(len(s) for s in satirlar)
        matris = np.full((len(satirlar), max_sutun), np.nan, dtype=np.float32)
        for i, s in enumerate(satirlar):
            matris[i, :len(s)] = s

        # NaN değerleri komşu ortalama ile doldur
        nan_maske = np.isnan(matris)
        if nan_maske.any():
            col_means = np.nanmean(matris, axis=0)
            col_means = np.where(np.isnan(col_means), 25.0, col_means)
            nan_idx = np.where(nan_maske)
            matris[nan_idx] = col_means[nan_idx[1]]

        ham_C = matris

        # Makul sıcaklık aralığı kontrolü
        if not (-50 < float(np.nanmean(ham_C)) < 500):
            print(f"Uyari: Ortalama sicaklik {np.nanmean(ham_C):.1f}C — "
                  f"birim kontrolu yapiniz.")

        # Görsel: normalize et → renkli
        norm = ((ham_C - ham_C.min()) /
                max(ham_C.max() - ham_C.min(), 0.01) * 255).astype(np.uint8)
        gorsel = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        meta = {
            't_min':  float(ham_C.min()),
            't_max':  float(ham_C.max()),
            't_ort':  float(ham_C.mean()),
            'boyut':  f"{ham_C.shape[1]}x{ham_C.shape[0]} piksel",
            'kaynak': f"Excel ({ham_C.shape[1]}x{ham_C.shape[0]})",
        }

        print(f"  Excel okundu: {ham_C.shape[1]}x{ham_C.shape[0]} piksel  "
              f"T_min={meta['t_min']:.2f}C  T_max={meta['t_max']:.2f}C")

        return gorsel, ham_C, meta


# ─── Termal İşaretleyici ─────────────────────────────────────────────────────

@dataclass
class IsaretleyiciDurum:
    palet_idx: int = 0
    esik_C: float = 40.0          # °C eşik alarmı (datasheet aralığına göre)
    esik_aktif: bool = True
    profil_aktif: bool = True
    super_res: bool = False        # SuperResolution: 320×240 mod
    delta_t_aktif: bool = False    # Delta T: iki nokta arası fark
    delta_nokta1: tuple = (0, 0)   # piksel koordinatı
    delta_nokta2: tuple = (0, 0)
    delta_secim: int = 0           # hangi nokta seçiliyor (0 veya 1)
    fare_x: int = 0
    fare_y: int = 0
    t_min: float = T_MIN_GERCEK
    t_max: float = T_MAX_GERCEK


def sicaklik_haritasi_olustur(gorsel_bgr: np.ndarray,
                               ham_C: np.ndarray | None,
                               durum: IsaretleyiciDurum) -> np.ndarray:
    """
    Giriş görüntüsünü termal renk paletine dönüştürür.
    Ham sıcaklık verisi varsa onu kullanır; yoksa gri tondan tahmin eder.
    """
    if ham_C is not None:
        # Gerçek sıcaklıkları 0-255'e normalize et
        t_min, t_max = ham_C.min(), ham_C.max()
        norm = ((ham_C - t_min) / max(t_max - t_min, 0.01) * 255
                ).astype(np.uint8)
    else:
        gri = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2GRAY)
        gri = cv2.GaussianBlur(gri, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm = clahe.apply(gri)

    _, cmap = PALETLER[durum.palet_idx]
    termal = cv2.applyColorMap(norm, cmap)
    return termal, norm


def sicak_soguk_isaretله(goruntu: np.ndarray,
                          norm: np.ndarray,
                          ham_C: np.ndarray | None,
                          durum: IsaretleyiciDurum) -> None:
    """
    Sıcak (max) ve soğuk (min) noktaları işaretler. (in-place)
    """
    if ham_C is not None:
        veri = ham_C
        def to_c(v): return v
    else:
        veri = norm.astype(np.float32)
        def to_c(v): return piksel_to_sicaklik(v, durum.t_min, durum.t_max)

    # Ölçeklenmiş koordinatlara dönüştürme için çarpan
    h_v, w_v = veri.shape[:2]
    h_g, w_g = goruntu.shape[:2]
    sx, sy = w_g / w_v, h_g / h_v

    # Sıcak nokta
    _, max_v, _, max_loc = cv2.minMaxLoc(veri)
    mx = int(max_loc[0] * sx)
    my = int(max_loc[1] * sy)
    t_max_val = to_c(max_v)
    cv2.circle(goruntu, (mx, my), 12, (0, 0, 255), 2)
    cv2.drawMarker(goruntu, (mx, my), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(goruntu, f"MAX {t_max_val:.1f}C",
                (mx + 14, my - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    # Soğuk nokta
    min_v, _, min_loc, _ = cv2.minMaxLoc(veri)
    mnx = int(min_loc[0] * sx)
    mny = int(min_loc[1] * sy)
    t_min_val = to_c(min_v)
    cv2.circle(goruntu, (mnx, mny), 12, (255, 100, 0), 2)
    cv2.drawMarker(goruntu, (mnx, mny), (255, 100, 0),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(goruntu, f"MIN {t_min_val:.1f}C",
                (mnx + 14, mny + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 100, 0), 2)


def delta_t_ciz(goruntu: np.ndarray,
                norm: np.ndarray,
                ham_C: np.ndarray | None,
                durum: IsaretleyiciDurum) -> None:
    """
    Delta T — iki nokta arası sıcaklık farkı (datasheet'te tanımlı ölçüm fonksiyonu).
    Sol tık: nokta 1 seç  |  Sağ tık: nokta 2 seç  |  [X] tuşu: aktif/pasif
    """
    H, W = goruntu.shape[:2]

    def piksel_sicaklik(px, py) -> float:
        """Ekran koordinatını sıcaklığa çevirir."""
        if ham_C is not None:
            hv, wv = ham_C.shape
            vx = int(np.clip(px / W * wv, 0, wv - 1))
            vy = int(np.clip(py / H * hv, 0, hv - 1))
            return float(ham_C[vy, vx])
        norm_b = cv2.resize(norm, (W, H))
        return piksel_to_sicaklik(float(norm_b[py, px]),
                                   durum.t_min, durum.t_max)

    renkler = [(0, 255, 0), (255, 165, 0)]   # Nokta1=yeşil, Nokta2=turuncu
    noktalar = [durum.delta_nokta1, durum.delta_nokta2]

    for i, (px, py) in enumerate(noktalar):
        if px == 0 and py == 0:
            continue
        t = piksel_sicaklik(px, py)
        cv2.drawMarker(goruntu, (px, py), renkler[i],
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(goruntu, f"P{i+1}: {t:.1f}C",
                    (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renkler[i], 2)

    # İki nokta da seçiliyse farkı göster
    p1x, p1y = durum.delta_nokta1
    p2x, p2y = durum.delta_nokta2
    if not (p1x == 0 and p1y == 0) and not (p2x == 0 and p2y == 0):
        t1 = piksel_sicaklik(p1x, p1y)
        t2 = piksel_sicaklik(p2x, p2y)
        delta = abs(t1 - t2)

        # Noktaları birleştiren çizgi
        cv2.line(goruntu, (p1x, p1y), (p2x, p2y), (200, 200, 200), 1,
                 cv2.LINE_AA)

        # Orta noktada Delta T etiketi
        mx_d = (p1x + p2x) // 2
        my_d = (p1y + p2y) // 2
        etiket = f"DT={delta:.1f}C (+-{DOGRULUK_C}C)"
        cv2.rectangle(goruntu, (mx_d - 2, my_d - 16),
                      (mx_d + len(etiket) * 8 + 2, my_d + 4),
                      (30, 30, 30), -1)
        cv2.putText(goruntu, etiket, (mx_d, my_d),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

    # Sol üst köşe: Delta T modu aktif uyarısı
    cv2.putText(goruntu, "[X]DeltaT: SOL=P1 SAG=P2",
                (8, goruntu.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def esik_maske_uygula(goruntu: np.ndarray,
                       norm: np.ndarray,
                       ham_C: np.ndarray | None,
                       durum: IsaretleyiciDurum) -> None:
    """
    Eşik değerinin üzerindeki bölgelere kırmızı şeffaf maske uygular. (in-place)
    """
    h_g, w_g = goruntu.shape[:2]

    if ham_C is not None:
        esik_norm = int((durum.esik_C - ham_C.min()) /
                        max(ham_C.max() - ham_C.min(), 0.01) * 255)
        kaynak = (
            (ham_C - ham_C.min()) /
            max(ham_C.max() - ham_C.min(), 0.01) * 255
        ).astype(np.uint8)
    else:
        esik_norm = sicaklik_to_piksel(durum.esik_C, durum.t_min, durum.t_max)
        kaynak = norm

    kaynak_buyuk = cv2.resize(kaynak, (w_g, h_g),
                               interpolation=cv2.INTER_LINEAR)
    _, maske = cv2.threshold(kaynak_buyuk, esik_norm, 255, cv2.THRESH_BINARY)

    # Kırmızı şeffaf katman
    kirmizi = np.zeros_like(goruntu)
    kirmizi[maske == 255] = (0, 0, 200)
    cv2.addWeighted(goruntu, 1.0, kirmizi, 0.35, 0, goruntu)

    # Kontur çiz
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(goruntu, konturlar, -1, (0, 0, 255), 1)


def profil_ciz(goruntu: np.ndarray,
               norm: np.ndarray,
               ham_C: np.ndarray | None,
               durum: IsaretleyiciDurum) -> None:
    """
    Ortadan yatay sıcaklık profili çizgisi ve altta mini grafik çizer. (in-place)
    """
    H, W = goruntu.shape[:2]
    satir_y = H // 2

    if ham_C is not None:
        h_v = ham_C.shape[0]
        satir_v = int(satir_y / H * h_v)
        satir_v = np.clip(satir_v, 0, h_v - 1)
        profil_raw = ham_C[satir_v, :]
        profil_norm = ((profil_raw - profil_raw.min()) /
                       max(profil_raw.max() - profil_raw.min(), 0.01)
                       * 255).astype(np.uint8)
        profil_C = profil_raw
    else:
        norm_byk = cv2.resize(norm, (W, H))
        profil_norm = norm_byk[satir_y, :]
        profil_C = np.array([piksel_to_sicaklik(v, durum.t_min, durum.t_max)
                             for v in profil_norm])

    # Çizgi
    cv2.line(goruntu, (0, satir_y), (W, satir_y), (255, 255, 0), 1)

    # Alt grafik arkaplanı
    grafik_h = 60
    grafik_y = H - grafik_h - 5
    ov = goruntu.copy()
    cv2.rectangle(ov, (0, grafik_y), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, goruntu, 0.45, 0, goruntu)

    # Profil çizgisi
    n = len(profil_C)
    noktalar = []
    for i, c in enumerate(profil_C):
        x = int(i / n * W)
        t_min_p, t_max_p = profil_C.min(), profil_C.max()
        aralik = max(t_max_p - t_min_p, 0.1)
        y = grafik_y + grafik_h - int((c - t_min_p) / aralik * (grafik_h - 8)) - 4
        noktalar.append((x, y))

    for i in range(len(noktalar) - 1):
        cv2.line(goruntu, noktalar[i], noktalar[i+1], (0, 255, 200), 1)

    # Min/Max etiket
    cv2.putText(goruntu, f"{profil_C.min():.1f}C",
                (2, H - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(goruntu, f"{profil_C.max():.1f}C",
                (2, grafik_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1)


def renk_cubugu_olustur(yukseklik: int, palet_id: int,
                         t_min: float, t_max: float) -> np.ndarray:
    """Sağ kenara yapıştırılacak renk çubuğu (40 px genişlik)."""
    cubuk = np.zeros((yukseklik, 40, 1), dtype=np.uint8)
    for i in range(yukseklik):
        cubuk[i, :, 0] = int(255 * (1 - i / yukseklik))
    renkli = cv2.applyColorMap(cubuk, palet_id)

    adim = yukseklik // 5
    for k in range(6):
        y_pos = k * adim
        sicaklik = t_max - k * (t_max - t_min) / 5
        cv2.putText(renkli, f"{sicaklik:.0f}", (2, y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return renkli


def fare_sicaklik_yaz(goruntu: np.ndarray,
                       norm: np.ndarray,
                       ham_C: np.ndarray | None,
                       durum: IsaretleyiciDurum) -> None:
    """Fare pozisyonundaki piksel sıcaklığını yazar."""
    H, W = goruntu.shape[:2]
    fx, fy = durum.fare_x, durum.fare_y
    if not (0 <= fx < W and 0 <= fy < H):
        return

    if ham_C is not None:
        h_v, w_v = ham_C.shape
        vx = int(fx / W * w_v)
        vy = int(fy / H * h_v)
        vx, vy = np.clip(vx, 0, w_v-1), np.clip(vy, 0, h_v-1)
        t = ham_C[vy, vx]
    else:
        norm_byk = cv2.resize(norm, (W, H))
        t = piksel_to_sicaklik(float(norm_byk[fy, fx]),
                                durum.t_min, durum.t_max)

    cv2.circle(goruntu, (fx, fy), 4, (255, 255, 255), -1)
    etiket = f"{t:.1f} C"
    cv2.putText(goruntu, etiket, (fx + 8, fy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def ust_serit_yaz(goruntu: np.ndarray, durum: IsaretleyiciDurum,
                   meta: dict, dosya_adi: str = "") -> None:
    """Üst bilgi şeridi — Testo 865 ölçüm bilgileri (in-place)."""
    W = goruntu.shape[1]
    ov = goruntu.copy()
    cv2.rectangle(ov, (0, 0), (W, 48), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.6, goruntu, 0.4, 0, goruntu)

    palet_adi = PALETLER[durum.palet_idx][0].split('(')[0].strip()
    t_min_s = meta.get('t_min', durum.t_min)
    t_max_s = meta.get('t_max', durum.t_max)
    kaynak  = meta.get('kaynak', '?')
    sr_yazi = "SuperRes:ACIK" if durum.super_res else "160x120"

    # Satır 1: ölçüm değerleri
    bilgi1 = (f"MIN:{t_min_s:.1f}C  MAX:{t_max_s:.1f}C  "
              f"Esik:{durum.esik_C:.1f}C  "
              f"Dogr:+-{DOGRULUK_C}C  {sr_yazi}")
    cv2.putText(goruntu, bilgi1, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Satır 2: kontroller
    bilgi2 = (f"Palet:{palet_adi}[1-4]  "
              f"[T/t]Esik  [E]Mask  [P]Profil  [X]DeltaT  [R]SuperRes  "
              f"[S]Kaydet  [Q]Cikis")
    cv2.putText(goruntu, bilgi2, (6, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 180, 140), 1)

    if dosya_adi and dosya_adi != "DEMO":
        cv2.putText(goruntu, os.path.basename(dosya_adi),
                    (W - 250, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (150, 220, 150), 1)


def tam_isaretleme(gorsel_bgr: np.ndarray,
                   ham_C: np.ndarray | None,
                   meta: dict,
                   durum: IsaretleyiciDurum,
                   dosya_adi: str = "") -> np.ndarray:
    """
    Tüm işaretleme adımlarını birleştiren ana fonksiyon.
    SuperResolution aktifse ham_C'yi 320×240'a bicubic ölçekler.
    """
    # SuperResolution: 160×120 → 320×240 bicubic interpolasyon
    if durum.super_res and ham_C is not None:
        ham_C = cv2.resize(ham_C,
                           TESTO_SUPER_RES,
                           interpolation=cv2.INTER_CUBIC)

    # Görüntü boyutu standardize
    gorsel_std = cv2.resize(gorsel_bgr, GOSTERIM_BOYUTU)
    if ham_C is not None:
        durum.t_min = float(ham_C.min())
        durum.t_max = float(ham_C.max())

    # Termal renk haritası
    termal, norm = sicaklik_haritasi_olustur(gorsel_std, ham_C, durum)

    # Eşik maskesi
    if durum.esik_aktif:
        esik_maske_uygula(termal, norm, ham_C, durum)

    # Sıcak / soğuk nokta
    sicak_soguk_isaretله(termal, norm, ham_C, durum)

    # Profil çizgisi
    if durum.profil_aktif:
        profil_ciz(termal, norm, ham_C, durum)

    # Delta T
    if durum.delta_t_aktif:
        delta_t_ciz(termal, norm, ham_C, durum)

    # Fare sıcaklık okuma
    fare_sicaklik_yaz(termal, norm, ham_C, durum)

    # Üst şerit
    ust_serit_yaz(termal, durum, meta, dosya_adi)

    # Renk çubuğu
    _, cmap = PALETLER[durum.palet_idx]
    cubuk = renk_cubugu_olustur(
        GOSTERIM_BOYUTU[1], cmap, durum.t_min, durum.t_max
    )
    return np.hstack([termal, cubuk])


# ─── Fare Geri Çağırması ─────────────────────────────────────────────────────

_durum_ref = None

def fare_geri_cagir(event, x, y, flags, param):
    global _durum_ref
    if _durum_ref is None:
        return
    _durum_ref.fare_x = x
    _durum_ref.fare_y = y

    if _durum_ref.delta_t_aktif:
        if event == cv2.EVENT_LBUTTONDOWN:
            _durum_ref.delta_nokta1 = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            _durum_ref.delta_nokta2 = (x, y)


# ─── Mod 1: Dosya İzleyici ────────────────────────────────────────────────────

def dosya_modu(klasor: str):
    """
    Belirtilen klasörü izler. Yeni .bmt / .jpg / .png gelince işler.
    Herhangi bir dosya yoksa demo görüntü oluşturur.
    """
    klasor = Path(klasor)
    klasor.mkdir(parents=True, exist_ok=True)

    durum = IsaretleyiciDurum()
    global _durum_ref
    _durum_ref = durum

    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PENCERE_ADI, GOSTERIM_BOYUTU[0] + 40, GOSTERIM_BOYUTU[1] + 10)
    cv2.setMouseCallback(PENCERE_ADI, fare_geri_cagir)

    islenecek_uzantilar = {'.bmt', '.jpg', '.jpeg', '.png', '.bmp',
                           '.xlsx', '.xls', '.csv'}
    islenmis = set()
    son_dosya = None
    son_gorsel = None
    son_ham_C  = None
    son_meta   = {}

    print(f"Klasor izleniyor: {klasor.resolve()}")
    print("Desteklenen: .bmt  .jpg  .xlsx  .csv")

    while True:
        # Klasördeki dosyaları tara
        dosyalar = sorted(
            [f for f in klasor.iterdir()
             if f.suffix.lower() in islenecek_uzantilar],
            key=lambda x: x.stat().st_mtime
        )

        for dosya in dosyalar:
            if str(dosya) not in islenmis:
                print(f"Yeni dosya: {dosya.name}")
                ext = dosya.suffix.lower()
                if ext == '.bmt':
                    g, h, m = TestoBMTOkuyucu.bmt_oku(str(dosya))
                elif ext in ('.xlsx', '.xls', '.csv'):
                    g, h, m = TestoBMTOkuyucu.excel_oku(str(dosya))
                else:
                    g, h, m = TestoBMTOkuyucu.jpg_oku(str(dosya))

                if g is not None:
                    son_gorsel, son_ham_C, son_meta = g, h, m
                    son_dosya = str(dosya)
                    print(f"  Islendi: T_min={m.get('t_min',0):.1f}C  "
                          f"T_max={m.get('t_max',0):.1f}C  "
                          f"Kaynak={m.get('kaynak','?')}")
                islenmis.add(str(dosya))

        # Görüntü yoksa demo oluştur
        if son_gorsel is None:
            son_gorsel, son_ham_C, son_meta = _demo_goruntu_olustur()
            son_dosya = "DEMO"

        gosterim = tam_isaretleme(
            son_gorsel, son_ham_C, son_meta, durum, son_dosya
        )
        cv2.imshow(PENCERE_ADI, gosterim)

        tus = cv2.waitKey(200) & 0xFF   # 200 ms bekle → dosya tarama
        if tus == ord('q'):
            break
        _klavye_isle(tus, durum, gosterim)

    cv2.destroyAllWindows()


def _demo_goruntu_olustur():
    """
    Testo 865 çıkışını simüle eden sentetik sıcaklık haritası oluşturur.
    Gerçek kamera dosyası yokken arayüzü test etmeye yarar.
    """
    h, w = TESTO_COZUNURLUK[1], TESTO_COZUNURLUK[0]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Sentetik sıcaklık alanı: birkaç Gauss tepesi
    ham_C = (
        22.0
        + 10.0 * np.exp(-((xx - 0.3)**2 + (yy - 0.4)**2) / 0.02)
        + 8.0  * np.exp(-((xx - 0.7)**2 + (yy - 0.6)**2) / 0.015)
        + 5.0  * np.exp(-((xx - 0.5)**2 + (yy - 0.2)**2) / 0.03)
        + np.random.normal(0, 0.3, (h, w))
    ).astype(np.float32)

    norm = ((ham_C - ham_C.min()) /
            (ham_C.max() - ham_C.min()) * 255).astype(np.uint8)
    gorsel = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    meta = {
        't_min': float(ham_C.min()),
        't_max': float(ham_C.max()),
        't_ort': float(ham_C.mean()),
        'kaynak': 'DEMO (sentetik)',
    }
    return gorsel, ham_C, meta


# ─── Mod 2: Canlı Kamera ─────────────────────────────────────────────────────

def canli_modu():
    """
    Kamera UVC (/dev/video*) olarak görünüyorsa gerçek zamanlı işler.
    Testo 865 Linux'ta genellikle /dev/video0 veya /dev/video2 olarak görünür.
    """
    kamera = None
    for idx in range(4):
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    print(f"Kamera bulundu: /dev/video{idx}")
                    kamera = cap
                    break
        if kamera:
            break

    if kamera is None:
        print("UVC kamera bulunamadi. Demo modu baslatiliyor...")
        _canli_demo_dongu()
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH,  TESTO_COZUNURLUK[0])
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, TESTO_COZUNURLUK[1])

    print("Kamera isinıyor...")
    for _ in range(30):
        kamera.read()

    durum = IsaretleyiciDurum()
    global _durum_ref
    _durum_ref = durum
    meta = {'kaynak': 'canli_UVC'}

    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PENCERE_ADI, GOSTERIM_BOYUTU[0] + 40, GOSTERIM_BOYUTU[1] + 10)
    cv2.setMouseCallback(PENCERE_ADI, fare_geri_cagir)
    kayit_dizin = "testo865_kayitlar"
    os.makedirs(kayit_dizin, exist_ok=True)

    while True:
        ret, frame = kamera.read()
        if not ret or frame is None:
            break

        gosterim = tam_isaretleme(frame, None, meta, durum)
        cv2.imshow(PENCERE_ADI, gosterim)

        tus = cv2.waitKey(1) & 0xFF
        if tus == ord('q'):
            break
        _klavye_isle(tus, durum, gosterim, kayit_dizin)

    kamera.release()
    cv2.destroyAllWindows()


def _canli_demo_dongu():
    """Kamera yoksa sentetik veriyle gerçek zamanlı demo çalıştırır."""
    durum = IsaretleyiciDurum()
    global _durum_ref
    _durum_ref = durum

    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PENCERE_ADI, GOSTERIM_BOYUTU[0] + 40, GOSTERIM_BOYUTU[1] + 10)
    cv2.setMouseCallback(PENCERE_ADI, fare_geri_cagir)
    kayit_dizin = "testo865_kayitlar"
    os.makedirs(kayit_dizin, exist_ok=True)

    t0 = time.time()
    while True:
        dt = time.time() - t0
        gorsel, ham_C, meta = _demo_goruntu_olustur()
        # Animasyon: sıcak nokta gezinir
        ham_C += 3.0 * np.sin(dt)

        gosterim = tam_isaretleme(gorsel, ham_C, meta, durum, "CANLI DEMO")
        cv2.imshow(PENCERE_ADI, gosterim)

        tus = cv2.waitKey(33) & 0xFF   # ~30 FPS
        if tus == ord('q'):
            break
        _klavye_isle(tus, durum, gosterim, kayit_dizin)

    cv2.destroyAllWindows()


# ─── Klavye Ortak ────────────────────────────────────────────────────────────

def _klavye_isle(tus: int, durum: IsaretleyiciDurum,
                 gosterim: np.ndarray, kayit_dizin: str = ".") -> None:
    if tus == ord('t'):
        durum.esik_C = min(durum.esik_C + 1.0, T_MAX_GERCEK)
        print(f"Esik: {durum.esik_C:.1f} C")
    elif tus == ord('T'):
        durum.esik_C = max(durum.esik_C - 1.0, T_MIN_GERCEK)
        print(f"Esik: {durum.esik_C:.1f} C")
    elif tus == ord('e'):
        durum.esik_aktif = not durum.esik_aktif
        print(f"Esik maskesi: {'ACIK' if durum.esik_aktif else 'KAPALI'}")
    elif tus == ord('p'):
        durum.profil_aktif = not durum.profil_aktif
        print(f"Profil: {'ACIK' if durum.profil_aktif else 'KAPALI'}")
    elif tus == ord('x'):
        durum.delta_t_aktif = not durum.delta_t_aktif
        durum.delta_nokta1 = (0, 0)
        durum.delta_nokta2 = (0, 0)
        print(f"Delta T: {'ACIK — Sol tik P1, Sag tik P2' if durum.delta_t_aktif else 'KAPALI'}")
    elif tus == ord('r'):
        durum.super_res = not durum.super_res
        mod = "320x240 (SuperResolution)" if durum.super_res else "160x120 (native)"
        print(f"Cozunurluk modu: {mod}")
    elif ord('1') <= tus <= ord('4'):       # Testo'nun 4 paleti
        durum.palet_idx = tus - ord('1')
        print(f"Palet: {PALETLER[durum.palet_idx][0]}")
    elif tus == ord('s'):
        dosya = os.path.join(
            kayit_dizin, f"testo865_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        cv2.imwrite(dosya, gosterim)
        print(f"Kaydedildi: {dosya}")


# ─── Giriş Noktası ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Testo 865 Termal Kamera Görüntü İşleme"
    )
    parser.add_argument(
        '--mod', choices=['dosya', 'canli'], default='canli',
        help="'dosya': BMT/JPEG klasör izle  |  'canli': UVC/webcam gerçek zamanlı"
    )
    parser.add_argument(
        '--klasor', default='testo865_girdi',
        help="Dosya modu için izlenecek klasör (varsayılan: testo865_girdi)"
    )
    args = parser.parse_args()

    print(__doc__)

    if args.mod == 'dosya':
        dosya_modu(args.klasor)
    else:
        canli_modu()


if __name__ == "__main__":
    main()
