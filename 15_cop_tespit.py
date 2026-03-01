"""
15 - Çöp / Terk Edilmiş Nesne Tespiti  (Optik Akış + MOG2)
===========================================================
PC kamerası ile yere bırakılan / terk edilen nesneleri tespit eder.

Temel Fikir:
  foreground_maskesi  AND  dusuk_optik_akis  =>  terk edilmis nesne

Adımlar:
  1. MOG2 arka plan çıkarma   -> ön plan maskesi  (yeni giren nesneler)
  2. Farneback yoğun optik akış -> akış büyüklük haritası
  3. Akış büyüklüğü DÜŞÜK  +  Ön plan maskesi YÜKSEK  ->  durağan ön plan
  4. Morfolojik temizleme    -> küçük gürültü blobları at
  5. Kontur analizi          -> şüpheli nesne bölgesi bul
  6. Zaman takibi            -> N saniye kalırsa ALARM

Kontroller:
  [r] : Arka planı ve referans kareyi sıfırla
  [d] : Debug görünümü (4 panel)
  [s] : Ekran görüntüsü kaydet
  [q] : Çıkış
"""

import cv2
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Optional

# ─── Ayarlar ─────────────────────────────────────────────────────────────────

AKIS_HAREKET_ESIK  = 1.5    # px/kare — bunun altı "hareketsiz" sayılır
BLOB_MIN_ALAN      = 800    # px²  — çok küçük blobları yoksay
BLOB_MAX_ALAN      = 40000  # px²  — çok büyük (kişi) blobları yoksay
SUPHELI_SURE       = 2.5    # sn  — bu kadar kalırsa sarı uyarı
ALARM_SURE         = 5.0    # sn  — bu kadar kalırsa kırmızı alarm
ESLESTIRME_ESIK    = 80     # px  — blob eşleştirme mesafesi
KAYIP_KARE_ESIK    = 15     # kare — blob bu kadar kaybolursa düşür

PENCERE_ADI = "Cop / Terk Nesne Tespiti  —  Goruntu Isleme"

# ─── Nesne Takibi ─────────────────────────────────────────────────────────────

@dataclass
class Blob:
    id: int
    cx: float
    cy: float
    alan: float
    bbox: tuple                        # (x, y, w, h)
    ilk_gorunum: float = field(default_factory=time.time)
    kayip: int = 0

    @property
    def sure(self) -> float:
        return time.time() - self.ilk_gorunum

    def guncelle(self, cx, cy, alan, bbox):
        self.cx, self.cy, self.alan, self.bbox = cx, cy, alan, bbox
        self.kayip = 0

    def durum(self) -> tuple[str, tuple]:
        if self.sure >= ALARM_SURE:
            return "!! COP !!", (0, 0, 255)
        if self.sure >= SUPHELI_SURE:
            return "Suphecli",  (0, 200, 255)
        return "Yeni",          (0, 220, 80)


class Takipci:
    def __init__(self):
        self.bloblar: dict[int, Blob] = {}
        self._id = 0

    def guncelle(self, olcumler: list[tuple]) -> dict[int, Blob]:
        """olcumler: [(cx, cy, alan, bbox), ...]"""
        eslestirildi = set()

        for cx, cy, alan, bbox in olcumler:
            en_yakin_id, en_yakin_d = None, ESLESTIRME_ESIK
            for bid, b in self.bloblar.items():
                if bid in eslestirildi:
                    continue
                d = ((cx - b.cx)**2 + (cy - b.cy)**2) ** 0.5
                if d < en_yakin_d:
                    en_yakin_d, en_yakin_id = d, bid

            if en_yakin_id is not None:
                self.bloblar[en_yakin_id].guncelle(cx, cy, alan, bbox)
                eslestirildi.add(en_yakin_id)
            else:
                self.bloblar[self._id] = Blob(self._id, cx, cy, alan, bbox)
                self._id += 1

        for bid in list(self.bloblar):
            if bid not in eslestirildi:
                self.bloblar[bid].kayip += 1
                if self.bloblar[bid].kayip > KAYIP_KARE_ESIK:
                    del self.bloblar[bid]

        return self.bloblar

    def sifirla(self):
        self.bloblar.clear()
        self._id = 0


# ─── Görüntü İşleme ─────────────────────────────────────────────────────────

def optik_akis_buyukluk(onceki_gri: np.ndarray,
                         simdiki_gri: np.ndarray) -> np.ndarray:
    """
    Farneback yoğun optik akış hesaplar.
    Her pikseldeki hareket vektörünün büyüklüğünü (magnitude) döndürür.
    """
    flow = cv2.calcOpticalFlowFarneback(
        onceki_gri, simdiki_gri,
        None,
        pyr_scale=0.5,   # piramit ölçeği
        levels=3,        # piramit katmanı
        winsize=13,      # pencere boyutu
        iterations=3,
        poly_n=5,        # polinomsal komşuluk
        poly_sigma=1.1,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag


def terk_maske_hesapla(on_plan: np.ndarray,
                        akis_mag: np.ndarray) -> np.ndarray:
    """
    Ön plan maskesi + düşük optik akış = terk edilmiş nesne maskesi.
    """
    # Hareketsiz piksel maskesi
    hareketsiz = (akis_mag < AKIS_HAREKET_ESIK).astype(np.uint8) * 255

    # İkisinin kesişimi
    terk = cv2.bitwise_and(on_plan, hareketsiz)

    # Morfolojik temizleme: küçük delikleri doldur, gürültüyü at
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    terk = cv2.morphologyEx(terk, cv2.MORPH_OPEN,  k3, iterations=2)
    terk = cv2.morphologyEx(terk, cv2.MORPH_CLOSE, k9, iterations=3)
    return terk


def konturlardan_olcum(terk_maske: np.ndarray) -> list[tuple]:
    """Konturları bul, alan ve bbox hesapla, boyut filtresini uygula."""
    konturlar, _ = cv2.findContours(
        terk_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    olcumler = []
    for k in konturlar:
        alan = cv2.contourArea(k)
        if not (BLOB_MIN_ALAN <= alan <= BLOB_MAX_ALAN):
            continue
        M = cv2.moments(k)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        bbox = cv2.boundingRect(k)
        olcumler.append((cx, cy, alan, bbox))
    return olcumler


# ─── Çizim ──────────────────────────────────────────────────────────────────

def blob_ciz(goruntu: np.ndarray, bloblar: dict[int, Blob]) -> int:
    """Blobları görüntü üzerine çizer. Alarm sayısını döndürür."""
    alarm_n = 0
    for bid, b in bloblar.items():
        if b.kayip > 3:
            continue
        etiket, renk = b.durum()
        x, y, w, h = b.bbox
        cx, cy = int(b.cx), int(b.cy)

        cv2.rectangle(goruntu, (x, y), (x+w, y+h), renk, 2)
        cv2.circle(goruntu, (cx, cy), 5, renk, -1)

        # Başlık
        cv2.putText(goruntu, f"#{bid} {etiket}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2)

        # Süre dolum çubuğu
        ilerleme = min(b.sure / ALARM_SURE, 1.0)
        bar_w = w
        dolu  = int(bar_w * ilerleme)
        cv2.rectangle(goruntu, (x, y + h + 3), (x + bar_w, y + h + 11),
                      (50, 50, 50), -1)
        cv2.rectangle(goruntu, (x, y + h + 3), (x + dolu, y + h + 11),
                      renk, -1)
        cv2.putText(goruntu, f"{b.sure:.1f}s / {ALARM_SURE:.0f}s",
                    (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    renk, 1)

        if b.sure >= ALARM_SURE:
            alarm_n += 1

    return alarm_n


def bilgi_seridi(goruntu: np.ndarray, blob_sayisi: int, alarm_n: int):
    """Üst şerit bilgisini yazar (in-place)."""
    h_g, w_g = goruntu.shape[:2]
    ov = goruntu.copy()
    cv2.rectangle(ov, (0, 0), (w_g, 36), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.6, goruntu, 0.4, 0, goruntu)

    cv2.putText(goruntu, f"Terk: {blob_sayisi} blob",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    if alarm_n:
        uyari = f"ALARM: {alarm_n} cop!"
        tw = cv2.getTextSize(uyari, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
        cv2.putText(goruntu, uyari,
                    (w_g // 2 - tw // 2, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(goruntu, "[R]Sifirla [D]Debug [S]Kaydet [Q]Cikis",
                (w_g - 340, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (130, 130, 130), 1)


def debug_panel_olustur(frame: np.ndarray,
                         on_plan: np.ndarray,
                         akis_mag: np.ndarray,
                         terk: np.ndarray) -> np.ndarray:
    """
    4 panelli debug görünümü:
      Sol üst  : orijinal kare
      Sağ üst  : ön plan maskesi (MOG2)
      Sol alt  : optik akış büyüklük haritası
      Sağ alt  : terk edilmiş nesne maskesi (kesişim)
    """
    H, W = frame.shape[:2]

    def bgr(gri):
        return cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)

    # Optik akışı görselleştir: büyük değer = parlak
    akis_norm = cv2.normalize(akis_mag, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)
    akis_renkli = cv2.applyColorMap(akis_norm, cv2.COLORMAP_JET)

    ust = np.hstack([frame, bgr(on_plan)])
    alt = np.hstack([akis_renkli, bgr(terk)])
    panel = np.vstack([ust, alt])

    # Panel etiketleri
    for (txt, px, py) in [
        ("Orijinal",            5,  15),
        ("On Plan (MOG2)",   W + 5,  15),
        ("Optik Akis (Mag)",     5, H + 15),
        ("Terk Maskesi",     W + 5, H + 15),
    ]:
        cv2.putText(panel, txt, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

    return panel


# ─── Kamera ─────────────────────────────────────────────────────────────────

def kamera_ac() -> Optional[cv2.VideoCapture]:
    for idx in range(3):
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    print(f"Kamera acildi: index={idx}")
                    return cap
                cap.release()
    return None


# ─── Ana Döngü ───────────────────────────────────────────────────────────────

def main():
    kamera = kamera_ac()
    if kamera is None:
        print("HATA: Kamera acilamadi.")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Kamera isinıyor...")
    for _ in range(30):
        kamera.read()

    # İlk kareyi al
    ret, frame = kamera.read()
    if not ret:
        print("HATA: Ilk kare alinamadi.")
        return

    # MOG2 — arka plan öğrenimi (gölge tespiti kapalı → temiz maske)
    mog = cv2.createBackgroundSubtractorMOG2(
        history=400, varThreshold=40, detectShadows=False
    )

    onceki_gri = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0),
                               cv2.COLOR_BGR2GRAY)
    takipci   = Takipci()
    debug_mod = False

    kayit_dizin = "cop_kayitlar"
    os.makedirs(kayit_dizin, exist_ok=True)

    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PENCERE_ADI, 1000, 580)

    print(__doc__)
    print("Hazir! Sahneye gisin, bir nesne birakip uzaklasin.")

    while True:
        ret, frame = kamera.read()
        if not ret or frame is None:
            break

        yumusak = cv2.GaussianBlur(frame, (5, 5), 0)
        simdiki_gri = cv2.cvtColor(yumusak, cv2.COLOR_BGR2GRAY)

        # ── Arka plan çıkarma ────────────────────────────────────────────────
        on_plan = mog.apply(yumusak)
        # Gölge piksellerini (127) temizle
        _, on_plan = cv2.threshold(on_plan, 200, 255, cv2.THRESH_BINARY)

        # ── Optik akış büyüklüğü ─────────────────────────────────────────────
        akis_mag = optik_akis_buyukluk(onceki_gri, simdiki_gri)

        # ── Terk edilmiş nesne maskesi ───────────────────────────────────────
        terk = terk_maske_hesapla(on_plan, akis_mag)

        # ── Kontur → ölçüm → takip ──────────────────────────────────────────
        olcumler = konturlardan_olcum(terk)
        bloblar  = takipci.guncelle(olcumler)

        # ── Çizim ────────────────────────────────────────────────────────────
        gosterim = frame.copy()
        alarm_n  = blob_ciz(gosterim, bloblar)
        bilgi_seridi(gosterim, len(olcumler), alarm_n)

        if debug_mod:
            panel = debug_panel_olustur(gosterim, on_plan, akis_mag, terk)
            cv2.imshow(PENCERE_ADI, panel)
        else:
            cv2.imshow(PENCERE_ADI, gosterim)

        onceki_gri = simdiki_gri.copy()

        # ── Klavye ──────────────────────────────────────────────────────────
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord('q'):
            break
        elif tus == ord('r'):
            mog = cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=40, detectShadows=False
            )
            takipci.sifirla()
            print("Sifirlandi.")
        elif tus == ord('d'):
            debug_mod = not debug_mod
            print(f"Debug: {'ACIK' if debug_mod else 'KAPALI'}")
        elif tus == ord('s'):
            dosya = os.path.join(
                kayit_dizin, f"cop_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(dosya, gosterim)
            print(f"Kaydedildi: {dosya}")

    kamera.release()
    cv2.destroyAllWindows()
    print("Program sonlandirildi.")


if __name__ == "__main__":
    main()
