"""
14 - Termal Kamera Simülasyonu (PC Webcam ile)
================================================
Gerçek zamanlı PC kamerası görüntüsünü termal kamera görüntüsüne dönüştürür.
Sıcak nokta (hot spot) tespiti, sıcaklık haritası ve renk paleti gösterimi içerir.

Kullanılan Teknikler:
- Gri ton dönüşümü + CLAHE (kontrast iyileştirme)
- OpenCV Colormap'leri (termal renk paletimleri)
- Morfolojik işlemler ile gürültü azaltma
- Kontur tespiti ile sıcak bölge işaretleme
- Gaussian blur ile düzleştirme

Kontroller:
  [1-7]  : Farklı termal palet seç
  [s]    : Ekran görüntüsü kaydet
  [h]    : Sıcak nokta tespitini aç/kapat
  [g]    : Izgara (grid) göster/gizle
  [q]    : Çıkış
  Not: Gerçek bir termal kamera olmadığından, webcam görüntüsünün parlaklık/yoğunluk değerleri sıcaklık gibi yorumlanıyor. Aydınlık bölgeler = "sıcak", karanlık bölgeler = "soğuk" olarak gösteriliyor. Bu ders için güzel bir simülasyon örneği olarak kullanılabilir.
"""

import cv2
import numpy as np
import time
import os

# ─── Sabitler ────────────────────────────────────────────────────────────────

PALETLER = {
    "1 - INFERNO  (varsayılan)": cv2.COLORMAP_INFERNO,
    "2 - JET      (klasik)    ": cv2.COLORMAP_JET,
    "3 - HOT      (sıcak)     ": cv2.COLORMAP_HOT,
    "4 - MAGMA    (volkanik)  ": cv2.COLORMAP_MAGMA,
    "5 - PLASMA   (plazma)    ": cv2.COLORMAP_PLASMA,
    "6 - RAINBOW  (gökkuşağı) ": cv2.COLORMAP_RAINBOW,
    "7 - OCEAN    (okyanus)   ": cv2.COLORMAP_OCEAN,
}
PALET_LISTESI = list(PALETLER.values())
PALET_ISIMLERI = list(PALETLER.keys())

SICAKLIK_MIN = 20   # °C (mavi taraf — soğuk)
SICAKLIK_MAX = 40   # °C (kırmızı taraf — sıcak)

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def goruntu_on_isle(frame: np.ndarray) -> np.ndarray:
    """
    Ham kamera görüntüsünü termal görüntü için hazırlar.
    Gri tona çevirip CLAHE ile kontrastı artırır, ardından Gaussian blur uygular.
    """
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE — lokal kontrast iyileştirme (termal detayları öne çıkarır)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gri = clahe.apply(gri)

    # Yüksek frekanslı gürültüyü yumuşat
    gri = cv2.GaussianBlur(gri, (9, 9), 0)
    return gri


def termal_uygula(gri: np.ndarray, palet_id: int) -> np.ndarray:
    """Gri ton görüntüye termal renk paleti uygular."""
    return cv2.applyColorMap(gri, palet_id)


def sicak_nokta_tespit(gri: np.ndarray, termal: np.ndarray,
                        esik_yuzde: float = 0.92) -> tuple[np.ndarray, list]:
    """
    Piksel yoğunluğu üst %8'de olan bölgeleri 'sıcak nokta' olarak işaretler.
    Morfolojik kapama ile küçük delikleri doldurur, ardından kontur bulur.

    Returns:
        isaretli  : Sıcak noktaların çizildiği termal kopya
        konturlar : Bulunan kontur listesi
    """
    esik_deger = int(gri.max() * esik_yuzde)
    _, maske = cv2.threshold(gri, esik_deger, 255, cv2.THRESH_BINARY)

    # Morfolojik kapama — küçük boşlukları doldur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    isaretli = termal.copy()
    for k in konturlar:
        alan = cv2.contourArea(k)
        if alan < 300:          # çok küçük bölgeleri yoksay
            continue

        x, y, w, h = cv2.boundingRect(k)
        # En sıcak piksel — bounding box içindeki maksimum yoğunluk noktası
        roi_gri = gri[y:y+h, x:x+w]
        min_v, max_v, _, max_loc = cv2.minMaxLoc(roi_gri)
        cx, cy = x + max_loc[0], y + max_loc[1]

        # Tahmini sıcaklık — piksel değerini [MIN, MAX]°C aralığına doğrusal eşle
        sicaklik = SICAKLIK_MIN + (max_v / 255) * (SICAKLIK_MAX - SICAKLIK_MIN)

        # Çizimler
        cv2.rectangle(isaretli, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.circle(isaretli, (cx, cy), 5, (255, 255, 255), -1)
        etiket = f"{sicaklik:.1f} C"
        cv2.putText(isaretli, etiket, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return isaretli, konturlar


def palet_cubugu_olustur(yukseklik: int, palet_id: int) -> np.ndarray:
    """
    Sağ kenara yapıştırılacak dikey renk çubuğu (color bar) oluşturur.
    Üst → sıcak (255), alt → soğuk (0).
    """
    cubuk = np.zeros((yukseklik, 40, 1), dtype=np.uint8)
    for i in range(yukseklik):
        cubuk[i, :, 0] = int(255 * (1 - i / yukseklik))
    cubuk_renkli = cv2.applyColorMap(cubuk, palet_id)

    # Üst ve alt etiketler
    cv2.putText(cubuk_renkli, f"{SICAKLIK_MAX}C", (2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(cubuk_renkli, f"{SICAKLIK_MIN}C", (2, yukseklik - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return cubuk_renkli


def izgara_ciz(goruntu: np.ndarray, satirlar: int = 4,
               sutunlar: int = 4) -> np.ndarray:
    """Görüntü üzerine şeffaf ızgara çizer; her hücrenin ortalama 'sıcaklığı' yazılır."""
    h, w = goruntu.shape[:2]
    kopyasi = goruntu.copy()

    for i in range(1, satirlar):
        y = int(i * h / satirlar)
        cv2.line(kopyasi, (0, y), (w, y), (100, 100, 100), 1)
    for j in range(1, sutunlar):
        x = int(j * w / sutunlar)
        cv2.line(kopyasi, (x, 0), (x, h), (100, 100, 100), 1)

    return kopyasi


def bilgi_yazdir(goruntu: np.ndarray, fps: float, palet_adi: str,
                 sicak_nokta_aktif: bool, izgara_aktif: bool) -> None:
    """Sol üst köşeye durum bilgisi yazar (in-place)."""
    h, w = goruntu.shape[:2]
    overlay = goruntu.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, goruntu, 0.55, 0, goruntu)

    satirlar = [
        f"FPS : {fps:5.1f}",
        f"Palet: {palet_adi.split('-')[1].strip().split(' ')[0]}",
        f"Sicak Nokta : {'ACIK [H]' if sicak_nokta_aktif else 'KAPALI [H]'}",
        f"Izgara      : {'ACIK [G]' if izgara_aktif else 'KAPALI [G]'}",
        "[1-7] Palet  [S] Kaydet  [Q] Cikis",
    ]
    for i, satir in enumerate(satirlar):
        renk = (0, 255, 200) if i < 4 else (180, 180, 180)
        cv2.putText(goruntu, satir, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 1)


# ─── Ana Döngü ────────────────────────────────────────────────────────────────

def kamera_ac() -> cv2.VideoCapture:
    """0, 1, 2 indekslerini sırayla dener; açılana kadar ilerler."""
    for idx in range(3):
        # Linux'ta V4L2 backend daha güvenilir
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"Kamera bulundu: index {idx}")
                return cap
            cap.release()
        # V4L2 çalışmazsa varsayılan backend'i dene
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"Kamera bulundu: index {idx} (varsayilan backend)")
                return cap
            cap.release()
    return None


def main():
    kamera = kamera_ac()
    if kamera is None:
        print("HATA: Hicbir kamera acilamadi. Kamera baglantisinizi kontrol edin.")
        return

    # Kamera çözünürlüğü
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Linux'ta kamera sensörü ilk ~30 karede ısınıyor — siyah kare sorunu
    print("Kamera isinıyor...")
    for _ in range(30):
        kamera.read()

    palet_idx       = 0          # başlangıç: INFERNO
    sicak_aktif     = True
    izgara_aktif    = False
    onceki_zaman    = time.time()
    fps             = 0.0

    kayit_dizin = "termal_kayitlar"
    os.makedirs(kayit_dizin, exist_ok=True)

    # Pencere boyutunu büyük aç
    pencere_adi = "Termal Kamera Simulasyonu — Goruntu Isleme"
    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pencere_adi, 1000, 600)

    print(__doc__)
    print("Kamera acildi. Pencereye tiklayin ve klavyeyi kullanin.")

    while True:
        ret, frame = kamera.read()
        if not ret or frame is None:
            print("HATA: Kamera karesi alinamadi.")
            break

        # ── FPS hesapla ──────────────────────────────────────────────────────
        simdi = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(simdi - onceki_zaman, 1e-9))
        onceki_zaman = simdi

        # ── Görüntü işleme boru hattı ────────────────────────────────────────
        gri    = goruntu_on_isle(frame)
        termal = termal_uygula(gri, PALET_LISTESI[palet_idx])

        if sicak_aktif:
            termal, _ = sicak_nokta_tespit(gri, termal)

        if izgara_aktif:
            termal = izgara_ciz(termal)

        bilgi_yazdir(termal, fps, PALET_ISIMLERI[palet_idx],
                     sicak_aktif, izgara_aktif)

        # ── Renk çubuğunu sağa yapıştır ──────────────────────────────────────
        cubuk = palet_cubugu_olustur(termal.shape[0], PALET_LISTESI[palet_idx])
        gosterim = np.hstack([termal, cubuk])

        cv2.imshow(pencere_adi, gosterim)

        # ── Klavye kontrolleri ────────────────────────────────────────────────
        tus = cv2.waitKey(1) & 0xFF

        if tus == ord('q'):
            break
        elif ord('1') <= tus <= ord('7'):
            palet_idx = tus - ord('1')
            print(f"Palet degistirildi: {PALET_ISIMLERI[palet_idx]}")
        elif tus == ord('h'):
            sicak_aktif = not sicak_aktif
            print(f"Sicak nokta tespiti: {'ACIK' if sicak_aktif else 'KAPALI'}")
        elif tus == ord('g'):
            izgara_aktif = not izgara_aktif
            print(f"Izgara: {'ACIK' if izgara_aktif else 'KAPALI'}")
        elif tus == ord('s'):
            dosya_adi = os.path.join(
                kayit_dizin,
                f"termal_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(dosya_adi, gosterim)
            print(f"Kaydedildi: {dosya_adi}")

    kamera.release()
    cv2.destroyAllWindows()
    print("Program sonlandirildi.")


if __name__ == "__main__":
    main()
