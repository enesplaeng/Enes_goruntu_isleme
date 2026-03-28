"""
Mini Photoshop - Goruntu Isleme Uygulamasi
Kamera | Kuantalama | Ornekleme | Kernel | Histogram | Kirpma | Akilli Silme
Gorunmezlik Pelerini (Invisibility Cloak)
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---- Renk Paleti ----
BG_DARK       = "#1a1a2e"
BG_PANEL      = "#16213e"
BG_CARD       = "#0f3460"
BG_CANVAS     = "#0d1117"
ACCENT        = "#e94560"
TEXT_PRIMARY   = "#e8e8e8"
TEXT_MUTED     = "#718096"
GREEN         = "#48bb78"
BLUE          = "#4299e1"
ORANGE        = "#ed8936"
PURPLE        = "#9f7aea"
CYAN          = "#38b2ac"
PINK          = "#d53f8c"
YELLOW        = "#ecc94b"
TEAL          = "#319795"
SLIDER_TROUGH = "#2d3748"
BORDER_COLOR  = "#2d3748"

FONT_TITLE   = ("Segoe UI", 13, "bold")
FONT_SECTION = ("Segoe UI", 10, "bold")
FONT_LABEL   = ("Segoe UI", 9)
FONT_BTN     = ("Segoe UI", 10, "bold")
FONT_STATUS  = ("Segoe UI", 8)
FONT_HERO    = ("Segoe UI", 16, "bold")
FONT_HERO_SUB= ("Segoe UI", 10)

FILTERS = [
    "Yok", "Gaussian Blur", "Median Blur",
    "Dilation (Genisletme)", "Erosion (Asindirma)",
    "Sobel (Kenar Bulma)", "Canny (Kenar Bulma)",
]

MODE_NORMAL = "normal"
MODE_CROP   = "crop"
MODE_ERASE  = "erase"

CAMERA_FPS_MS = 33


def make_button(parent, text, command, bg_color, fg_color="white", w=28):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg_color, fg=fg_color,
        activebackground=bg_color, activeforeground=fg_color,
        font=FONT_BTN, width=w, relief="flat", cursor="hand2",
        bd=0, highlightthickness=0, pady=6
    )
    original_bg = bg_color
    def on_enter(e):
        btn.configure(bg=_lighter(original_bg))
    def on_leave(e):
        btn.configure(bg=original_bg)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def _lighter(hex_color, amount=30):
    hex_color = hex_color.lstrip("#")
    r = min(255, int(hex_color[0:2], 16) + amount)
    g = min(255, int(hex_color[2:4], 16) + amount)
    b = min(255, int(hex_color[4:6], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


class PhotoshopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop")
        self.root.geometry("1300x830")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BG_DARK)

        # Goruntu verileri
        self.original_img = None
        self.working_img = None
        self.processed_img = None
        self.backup_img = None
        self.image_path = None
        self.max_display = (660, 440)

        # Kamera
        self.cap = None
        self.camera_active = False
        self._after_id = None

        # Mod & araclar
        self.mode = MODE_NORMAL
        self._crop_rect = None
        self._crop_start = None
        self._crop_end = None

        # Akilli silme icin maske
        self.erase_mask = None  # Tek kanal, ayni boyut, 0=dokunma, 255=sil

        # Gorunmezlik pelerini
        self.cloak_active = False
        self.cloak_bg = None  # Yakalanan arka plan karesi
        
        # Yuz algilama (Pelerin maskesini yuze-saca dokundurmamak icin)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None

        # Goruntuleme olcek bilgisi
        self._disp_scale = 1.0
        self._disp_offset_x = 0
        self._disp_offset_y = 0
        self._disp_w = 0
        self._disp_h = 0

        # ttk tema
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=BG_CARD, background=BG_CARD,
                        foreground=TEXT_PRIMARY, arrowcolor=ACCENT, borderwidth=0)
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", BG_CARD)],
                  foreground=[("readonly", TEXT_PRIMARY)])

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================
    # UI
    # ================================================================
    def _build_ui(self):
        # -- Ust baslik --
        header = tk.Frame(self.root, bg=BG_PANEL, height=44)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="  Mini Photoshop",
                 font=FONT_TITLE, bg=BG_PANEL, fg=ACCENT).pack(side=tk.LEFT, padx=12)

        self.lbl_mode = tk.Label(header, text="Mod: Normal",
                                  font=FONT_LABEL, bg=BG_PANEL, fg=GREEN)
        self.lbl_mode.pack(side=tk.RIGHT, padx=16)

        tk.Label(header, text="Kamera | Filtre | Kirpma | Akilli Silme | Histogram",
                 font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=8)

        # -- Alt durum cubugu --
        self.status_bar = tk.Label(
            self.root,
            text="  Hazir  |  Fotograf yukleyin veya kamerayi baslatin",
            bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_STATUS, anchor="w", padx=12, pady=4
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # -- Ana govde --
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        # Sol bolum
        left_frame = tk.Frame(body, bg=BG_DARK)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        # Goruntu alani - Canvas
        self.frame_canvas = tk.Frame(left_frame, bg=BG_CANVAS,
                                     highlightbackground=BORDER_COLOR, highlightthickness=1)
        self.frame_canvas.pack(fill=tk.BOTH, expand=True)

        self.img_canvas = tk.Canvas(self.frame_canvas, bg=BG_CANVAS,
                                     highlightthickness=0, cursor="crosshair")
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        self.img_canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.img_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.img_canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self._canvas_img_id = None
        self._show_placeholder()

        # Histogram alani
        hist_frame = tk.Frame(left_frame, bg=BG_DARK, height=180)
        hist_frame.pack(fill=tk.X, pady=(6, 0))
        hist_frame.pack_propagate(False)
        self._setup_histogram(hist_frame)

        # Sag: kontrol paneli
        panel_outer = tk.Frame(body, bg=BG_PANEL, width=300,
                               highlightbackground=BORDER_COLOR, highlightthickness=1)
        panel_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        panel_outer.pack_propagate(False)

        self._scroll_canvas = tk.Canvas(panel_outer, bg=BG_PANEL, highlightthickness=0, width=280)
        scrollbar = tk.Scrollbar(panel_outer, orient="vertical", command=self._scroll_canvas.yview)
        self.panel = tk.Frame(self._scroll_canvas, bg=BG_PANEL)

        self.panel.bind("<Configure>",
                        lambda e: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all")))
        self._scroll_canvas.create_window((0, 0), window=self.panel, anchor="nw")
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_panel()

    # ================================================================
    # Histogram
    # ================================================================
    def _setup_histogram(self, parent):
        self.hist_fig = Figure(figsize=(8, 1.6), dpi=80)
        self.hist_fig.patch.set_facecolor(BG_DARK)

        self.ax_r = self.hist_fig.add_subplot(1, 3, 1)
        self.ax_g = self.hist_fig.add_subplot(1, 3, 2)
        self.ax_b = self.hist_fig.add_subplot(1, 3, 3)

        for ax, title, color in [(self.ax_r, "Red", "#ff6b6b"),
                                  (self.ax_g, "Green", "#51cf66"),
                                  (self.ax_b, "Blue", "#339af0")]:
            ax.set_facecolor(BG_CANVAS)
            ax.set_title(title, fontsize=8, color=color, pad=3)
            ax.tick_params(labelsize=5, colors=TEXT_MUTED)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["bottom", "left"]:
                ax.spines[s].set_color(BORDER_COLOR)

        self.hist_fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.12, wspace=0.25)
        self.hist_canvas_mpl = FigureCanvasTkAgg(self.hist_fig, master=parent)
        self.hist_canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_histogram(self, img_bgr):
        if img_bgr is None:
            return
        b_ch, g_ch, r_ch = cv2.split(img_bgr)
        for ax, ch, clr in [(self.ax_r, r_ch, "#ff6b6b"),
                              (self.ax_g, g_ch, "#51cf66"),
                              (self.ax_b, b_ch, "#339af0")]:
            ax.clear()
            ax.set_facecolor(BG_CANVAS)
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
            ax.fill_between(range(256), hist, alpha=0.45, color=clr)
            ax.plot(hist, color=clr, linewidth=0.7)
            ax.set_xlim([0, 255])
            ax.tick_params(labelsize=5, colors=TEXT_MUTED)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["bottom", "left"]:
                ax.spines[s].set_color(BORDER_COLOR)
        for ax, (t, c) in zip([self.ax_r, self.ax_g, self.ax_b],
                               [("Red", "#ff6b6b"), ("Green", "#51cf66"), ("Blue", "#339af0")]):
            ax.set_title(t, fontsize=8, color=c, pad=3)
        self.hist_canvas_mpl.draw_idle()

    def _clear_histogram(self):
        for ax in [self.ax_r, self.ax_g, self.ax_b]:
            ax.clear()
            ax.set_facecolor(BG_CANVAS)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["bottom", "left"]:
                ax.spines[s].set_color(BORDER_COLOR)
        for ax, (t, c) in zip([self.ax_r, self.ax_g, self.ax_b],
                               [("Red", "#ff6b6b"), ("Green", "#51cf66"), ("Blue", "#339af0")]):
            ax.set_title(t, fontsize=8, color=c, pad=3)
        self.hist_canvas_mpl.draw_idle()

    # ================================================================
    # Placeholder
    # ================================================================
    def _show_placeholder(self):
        self.img_canvas.delete("all")
        self._canvas_img_id = None
        cw = self.img_canvas.winfo_width() or 660
        ch = self.img_canvas.winfo_height() or 440
        self.img_canvas.create_text(cw // 2, ch // 2 - 15,
                                     text="Fotograf yuklenmedi",
                                     font=FONT_HERO, fill=TEXT_MUTED)
        self.img_canvas.create_text(cw // 2, ch // 2 + 20,
                                     text="Fotograf yukleyin veya kamerayi baslatin",
                                     font=FONT_HERO_SUB, fill=TEXT_MUTED)

    # ================================================================
    # Kontrol Paneli
    # ================================================================
    def _build_panel(self):
        p = self.panel

        # -- Kamera --
        self._section(p, "Kamera (Canli)")
        self.btn_cam = make_button(p, "Kamerayi Baslat", self.toggle_camera, PURPLE)
        self.btn_cam.pack(pady=4, padx=10)
        self._sep(p)

        # -- Gorunmezlik Pelerini --
        self._section(p, "Gorunmezlik Pelerini")
        self._hint(p, "Arka plan hafizada tutulur,")
        self._hint(p, "SIYAH montu gosterdiginiz an silinir.")

        self.btn_cloak = make_button(p, "Pelerini Ac", self._toggle_cloak, TEAL, fg_color="white")
        self.btn_cloak.pack(pady=4, padx=10)

        self._hint(p, "--- SIYAH Mont HSV Ayari ---")
        self._hint(p, "Hue min / max (siyah=herhangi)")

        hsv_row1 = tk.Frame(p, bg=BG_PANEL)
        hsv_row1.pack(fill=tk.X, padx=16)
        self.scale_h_lo = self._slider(hsv_row1, 0, 179, live=False)
        self.scale_h_lo.set(0)
        self.scale_h_lo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.scale_h_hi = self._slider(hsv_row1, 0, 179, live=False)
        self.scale_h_hi.set(179)
        self.scale_h_hi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self._hint(p, "Saturation min / max")
        hsv_row2 = tk.Frame(p, bg=BG_PANEL)
        hsv_row2.pack(fill=tk.X, padx=16)
        self.scale_s_lo = self._slider(hsv_row2, 0, 255, live=False)
        self.scale_s_lo.set(0)
        self.scale_s_lo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.scale_s_hi = self._slider(hsv_row2, 0, 255, live=False)
        self.scale_s_hi.set(255)
        self.scale_s_hi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self._hint(p, "Value min / max (SIYAH = dusuk V!)")
        hsv_row3 = tk.Frame(p, bg=BG_PANEL)
        hsv_row3.pack(fill=tk.X, padx=16)
        self.scale_v_lo = self._slider(hsv_row3, 0, 255, live=False)
        self.scale_v_lo.set(0)
        self.scale_v_lo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.scale_v_hi = self._slider(hsv_row3, 0, 255, live=False)
        self.scale_v_hi.set(85)
        self.scale_v_hi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self._sep(p)

        # -- Dosya --
        self._section(p, "Dosya Islemleri")
        make_button(p, "Fotograf Yukle", self.load_image, GREEN).pack(pady=4, padx=10)
        make_button(p, "Fotografi Kaydet", self.save_image, BLUE).pack(pady=4, padx=10)
        self._sep(p)

        # -- Kirpma --
        self._section(p, "Kirpma (Crop)")
        self._hint(p, "Resmin uzerinde mouse ile alan secin")
        self.btn_crop_mode = make_button(p, "Kirpma Modunu Ac", self._toggle_crop_mode, CYAN)
        self.btn_crop_mode.pack(pady=4, padx=10)
        make_button(p, "Secimi Kirp", self._apply_crop, _lighter(CYAN, 20)).pack(pady=4, padx=10)
        self._sep(p)

        # -- Akilli Silme --
        self._section(p, "Akilli Silme (Inpainting)")
        self._hint(p, "Silmek istediginiz alani boyayin")
        self._hint(p, "Kirmizi ile isaretlenen alan silinecek")

        self.btn_erase_mode = make_button(p, "Silme Modunu Ac", self._toggle_erase_mode, PINK)
        self.btn_erase_mode.pack(pady=4, padx=10)

        self._hint(p, "Firca boyutu")
        self.scale_brush = self._slider(p, 5, 100, resolution=1, live=False)
        self.scale_brush.set(25)
        self.scale_brush.pack(pady=(0, 4), padx=16, fill=tk.X)

        self._hint(p, "Inpaint yaricapi (buyuk=daha puruzsuz)")
        self.scale_inpaint_r = self._slider(p, 1, 30, resolution=1, live=False)
        self.scale_inpaint_r.set(10)
        self.scale_inpaint_r.pack(pady=(0, 4), padx=16, fill=tk.X)

        self._hint(p, "Inpaint yontemi")
        self.inpaint_method_var = tk.StringVar(value="TELEA")
        im = ttk.Combobox(p, textvariable=self.inpaint_method_var,
                          values=["TELEA", "Navier-Stokes"], state="readonly",
                          width=30, style="Dark.TCombobox")
        im.pack(pady=4, padx=16)

        make_button(p, ">> Akilli Doldur <<", self._apply_inpaint, YELLOW, fg_color="#1a1a2e").pack(pady=6, padx=10)
        make_button(p, "Maskeyi Temizle", self._clear_mask, TEXT_MUTED, fg_color="white").pack(pady=2, padx=10)
        self._sep(p)

        # -- Ornekleme --
        self._section(p, "Ornekleme (Sampling)")
        self._hint(p, "Piksellestirme (1 = orijinal)")
        self.scale_samp = self._slider(p, 1, 30)
        self.scale_samp.set(1)
        self.scale_samp.pack(pady=(0, 8), padx=16, fill=tk.X)

        # -- Kuantalama --
        self._section(p, "Kuantalama (Quantization)")
        self._hint(p, "Bit derinligi (8 = orijinal)")
        self.scale_quant = self._slider(p, 8, 1)
        self.scale_quant.set(8)
        self.scale_quant.pack(pady=(0, 8), padx=16, fill=tk.X)
        self._sep(p)

        # -- Filtreler --
        self._section(p, "Kernel & Filtreler")
        self.filter_var = tk.StringVar(value="Yok")
        combo = ttk.Combobox(p, textvariable=self.filter_var,
                             values=FILTERS, state="readonly",
                             width=30, style="Dark.TCombobox")
        combo.pack(pady=6, padx=16)
        combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self._hint(p, "Kernel boyutu (tek sayi)")
        self.scale_ksize = self._slider(p, 3, 31, resolution=2)
        self.scale_ksize.set(3)
        self.scale_ksize.pack(pady=(0, 8), padx=16, fill=tk.X)
        self._sep(p)

        # -- Sifirlama --
        self._section(p, "Sifirlama")
        make_button(p, "Geri Al (Son Islem)", self._undo, ORANGE).pack(pady=4, padx=10)
        make_button(p, "Tumunu Sifirla", self.reset_all, ORANGE).pack(pady=4, padx=10)
        make_button(p, "Resmi Kaldir", self.clear_image, ACCENT).pack(pady=4, padx=10)

        tk.Frame(p, bg=BG_PANEL, height=30).pack()

    # -- Panel yardimcilari --
    def _section(self, parent, text):
        tk.Label(parent, text=text, font=FONT_SECTION,
                 bg=BG_PANEL, fg=ACCENT, anchor="w").pack(fill=tk.X, padx=14, pady=(14, 2))

    def _hint(self, parent, text):
        tk.Label(parent, text=text, font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED, anchor="w").pack(fill=tk.X, padx=18, pady=(0, 4))

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER_COLOR, height=1).pack(fill=tk.X, padx=14, pady=10)

    def _slider(self, parent, from_, to, resolution=1, live=True):
        cmd = self._on_slider_change if live else None
        return tk.Scale(parent, from_=from_, to=to, resolution=resolution,
                        orient=tk.HORIZONTAL,
                        bg=BG_PANEL, fg=TEXT_PRIMARY,
                        troughcolor=SLIDER_TROUGH,
                        activebackground=ACCENT,
                        highlightthickness=0, bd=0,
                        sliderrelief="flat",
                        font=FONT_LABEL,
                        command=cmd)

    def _on_slider_change(self, *_):
        if not self.camera_active:
            self.update_image()

    def _on_filter_change(self, *_):
        if not self.camera_active:
            self.update_image()

    # ================================================================
    # Mod Yonetimi
    # ================================================================
    def _set_mode(self, mode):
        self.mode = mode
        if mode == MODE_NORMAL:
            self.lbl_mode.configure(text="Mod: Normal", fg=GREEN)
            self.img_canvas.configure(cursor="crosshair")
            self.btn_crop_mode.configure(text="Kirpma Modunu Ac", bg=CYAN)
            self.btn_erase_mode.configure(text="Silme Modunu Ac", bg=PINK)
        elif mode == MODE_CROP:
            self.lbl_mode.configure(text="Mod: KIRPMA", fg=CYAN)
            self.img_canvas.configure(cursor="cross")
            self.btn_crop_mode.configure(text="Kirpma Modunu Kapat", bg=ACCENT)
            self.btn_erase_mode.configure(text="Silme Modunu Ac", bg=PINK)
            self._status("Kirpma modu  |  Resmin uzerinde dikdortgen cizin")
        elif mode == MODE_ERASE:
            self.lbl_mode.configure(text="Mod: AKILLI SILME", fg=PINK)
            self.img_canvas.configure(cursor="circle")
            self.btn_crop_mode.configure(text="Kirpma Modunu Ac", bg=CYAN)
            self.btn_erase_mode.configure(text="Silme Modunu Kapat", bg=ACCENT)
            self._init_mask()
            self._status("Akilli silme modu  |  Silmek istediginiz alani boyayin, sonra 'Akilli Doldur' basin")

    def _toggle_crop_mode(self):
        if self.mode == MODE_CROP:
            self._clear_crop_rect()
            self._set_mode(MODE_NORMAL)
        else:
            self._set_mode(MODE_CROP)

    def _toggle_erase_mode(self):
        if self.mode == MODE_ERASE:
            self._set_mode(MODE_NORMAL)
            self.update_image()  # Maske overlay'i kaldir
        else:
            self._set_mode(MODE_ERASE)

    # ================================================================
    # Mouse Olaylari
    # ================================================================
    def _on_mouse_down(self, event):
        if self.working_img is None:
            return
        if self.mode == MODE_CROP:
            self._clear_crop_rect()
            self._crop_start = (event.x, event.y)
            self._crop_end = (event.x, event.y)
        elif self.mode == MODE_ERASE:
            self._paint_mask(event.x, event.y)

    def _on_mouse_drag(self, event):
        if self.working_img is None:
            return
        if self.mode == MODE_CROP and self._crop_start is not None:
            self._crop_end = (event.x, event.y)
            self._draw_crop_rect()
        elif self.mode == MODE_ERASE:
            self._paint_mask(event.x, event.y)

    def _on_mouse_up(self, event):
        if self.mode == MODE_CROP and self._crop_start is not None:
            self._crop_end = (event.x, event.y)
            self._draw_crop_rect()
            self._status("Alan secildi  |  'Secimi Kirp' butonuna basin")

    # ================================================================
    # Kirpma (Crop)
    # ================================================================
    def _draw_crop_rect(self):
        if self._crop_rect is not None:
            self.img_canvas.delete(self._crop_rect)
        x1, y1 = self._crop_start
        x2, y2 = self._crop_end
        self._crop_rect = self.img_canvas.create_rectangle(
            x1, y1, x2, y2, outline="#00ff88", width=2, dash=(6, 4)
        )

    def _clear_crop_rect(self):
        if self._crop_rect is not None:
            self.img_canvas.delete(self._crop_rect)
            self._crop_rect = None
        self._crop_start = None
        self._crop_end = None

    def _apply_crop(self):
        if self.working_img is None:
            messagebox.showwarning("Uyari", "Once bir goruntu yukleyin!")
            return
        if self._crop_start is None or self._crop_end is None:
            messagebox.showwarning("Uyari", "Once mouse ile bir alan secin!")
            return

        self._save_backup()

        ix1, iy1 = self._canvas_to_img(self._crop_start[0], self._crop_start[1])
        ix2, iy2 = self._canvas_to_img(self._crop_end[0], self._crop_end[1])
        x1, x2 = sorted([ix1, ix2])
        y1, y2 = sorted([iy1, iy2])

        h, w = self.working_img.shape[:2]
        x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h))

        if x2-x1 < 5 or y2-y1 < 5:
            messagebox.showwarning("Uyari", "Secilen alan cok kucuk!")
            return

        self.working_img = self.working_img[y1:y2, x1:x2].copy()
        self.erase_mask = None
        self._clear_crop_rect()
        self._set_mode(MODE_NORMAL)
        self._status(f"Kirpildi  |  Yeni boyut: {x2-x1}x{y2-y1}")
        self.update_image()

    # ================================================================
    # Akilli Silme (Inpainting)
    # ================================================================
    def _init_mask(self):
        """Maske yoksa veya boyutu uyusmuyorsa yeni olustur."""
        if self.working_img is None:
            return
        h, w = self.working_img.shape[:2]
        if self.erase_mask is None or self.erase_mask.shape[:2] != (h, w):
            self.erase_mask = np.zeros((h, w), dtype=np.uint8)

    def _paint_mask(self, cx, cy):
        """Mouse konumunda maskeye beyaz daire cizer ve overlay gosterir."""
        if self.working_img is None or self.erase_mask is None:
            return

        ix, iy = self._canvas_to_img(cx, cy)
        brush = self.scale_brush.get()
        img_brush = max(2, int(brush / self._disp_scale)) if self._disp_scale > 0 else brush

        cv2.circle(self.erase_mask, (ix, iy), img_brush, 255, -1)

        # Kirmizi overlay ile goster
        self._display_with_mask()

    def _display_with_mask(self):
        """Calisma goruntusunu maske overlay ile gosterir."""
        if self.working_img is None:
            return

        vis = self.working_img.copy()

        if self.erase_mask is not None and np.any(self.erase_mask):
            # Kirmizi yari saydam overlay
            overlay = vis.copy()
            overlay[self.erase_mask > 0] = [0, 0, 255]  # BGR kirmizi
            cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

            # Maske sinirini beyaz cizgiyle goster
            contours, _ = cv2.findContours(self.erase_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        self._display(vis)

    def _apply_inpaint(self):
        """Maskelenmis alani akilli doldurma (inpainting) ile siler."""
        if self.working_img is None:
            messagebox.showwarning("Uyari", "Once bir goruntu yukleyin!")
            return
        if self.erase_mask is None or not np.any(self.erase_mask):
            messagebox.showwarning("Uyari", "Once silmek istediginiz alani boyayin!\nSilme modunu acip resmin uzerinde mouse ile cizin.")
            return

        self._save_backup()

        radius = self.scale_inpaint_r.get()
        method_name = self.inpaint_method_var.get()
        method = cv2.INPAINT_TELEA if method_name == "TELEA" else cv2.INPAINT_NS

        self._status("Inpainting isleniyor... Lutfen bekleyin")
        self.root.update_idletasks()

        # Inpainting uygula
        result = cv2.inpaint(self.working_img, self.erase_mask, radius, method)

        self.working_img = result
        self.erase_mask = np.zeros_like(self.erase_mask)

        self._set_mode(MODE_NORMAL)
        self._status(f"Akilli silme tamamlandi  |  Yontem: {method_name}, Yaricap: {radius}")
        self.update_image()

    def _clear_mask(self):
        """Maskeyi temizler."""
        if self.working_img is not None:
            h, w = self.working_img.shape[:2]
            self.erase_mask = np.zeros((h, w), dtype=np.uint8)
            if self.mode == MODE_ERASE:
                self._display_with_mask()
            else:
                self.update_image()
            self._status("Maske temizlendi")

    # ================================================================
    # Koordinat Donusumu
    # ================================================================
    def _canvas_to_img(self, cx, cy):
        if self._disp_scale == 0:
            return 0, 0
        ix = int((cx - self._disp_offset_x) / self._disp_scale)
        iy = int((cy - self._disp_offset_y) / self._disp_scale)
        if self.working_img is not None:
            h, w = self.working_img.shape[:2]
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))
        return ix, iy

    # ================================================================
    # Kamera
    # ================================================================
    def toggle_camera(self):
        if self.camera_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self._set_mode(MODE_NORMAL)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Kamera acilamadi!")
            self.cap = None
            return
        self.camera_active = True
        self.btn_cam.configure(text="Kamerayi Durdur", bg=ACCENT)
        self._status("Kamera aktif  |  Canli goruntu isleniyor")
        self._camera_loop()

    def _stop_camera(self):
        self.camera_active = False
        self.cloak_active = False
        self.btn_cloak.configure(text="Pelerini Ac", bg=TEAL)
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(text="Kamerayi Baslat", bg=PURPLE)
        self._status("Kamera durduruldu  |  Son kare uzerinde calisabilirsiniz")

    def _camera_loop(self):
        if not self.camera_active or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.original_img = frame
            # Pelerin kapaliyken mevcut arka plani hafizada tut
            if not self.cloak_active:
                self.cloak_bg = frame.copy()
            # Gorunmezlik pelerini aktifse uygula
            elif self.cloak_active and self.cloak_bg is not None:
                frame = self._apply_cloak(frame)
            self.working_img = frame.copy()
            self.update_image()
        self._after_id = self.root.after(CAMERA_FPS_MS, self._camera_loop)

    # ================================================================
    # Gorunmezlik Pelerini (Invisibility Cloak)
    # ================================================================
    def _toggle_cloak(self):
        """Gorunmezlik pelerini modunu acar/kapatir."""
        if not self.camera_active:
            messagebox.showwarning("Uyari", "Once kamerayi baslatin!")
            return

        self.cloak_active = not self.cloak_active
        if self.cloak_active:
            self.btn_cloak.configure(text="Pelerini Kapat", bg=ACCENT)
            self._status("Gorunmezlik pelerini AKTIF  |  Siyah montunuzu gosterin!")
        else:
            self.btn_cloak.configure(text="Pelerini Ac", bg=TEAL)
            self._status("Gorunmezlik pelerini kapatildi")

    def _apply_cloak(self, frame):
        """Mont rengini tespit edip arka plani canli gunceller ve degistirir."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Slider'lardan HSV araligini al
        h_lo = self.scale_h_lo.get()
        s_lo = self.scale_s_lo.get()
        v_lo = self.scale_v_lo.get()
        h_hi = self.scale_h_hi.get()
        s_hi = self.scale_s_hi.get()
        v_hi = self.scale_v_hi.get()

        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])

        # Renk maskesi olustur
        mask = cv2.inRange(hsv, lower, upper)
        # Morfolojik islemler ile maskeyi temizle
        # Kullanicinin istedigi sira: Genisletme, Asindirma, Acma, Kapama
        kernel = np.ones((5, 5), np.uint8)

        # 1. Genisletme (Dilation) -> Algilanan parcalari buyut
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 2. Asindirma (Erosion) -> Kucuk gurultuleri torpule
        mask = cv2.erode(mask, kernel, iterations=2)
        
        # 3. Acma (Opening) -> Disaridaki bagimsiz benekleri sil
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 4. Kapama (Closing) -> Mont icinde kalan delikleri, siyah noktalari kapat
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # GaussianBlur ile kenar gecislerini yumusat (dogal gorunum)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        # ── YÜZ VE SAÇ KORUMASI ──
        # Siyah saclarin veya gozlerin maskeye dahil olup gorunmez olmasini engelle
        if hasattr(self, 'face_cascade') and self.face_cascade is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Yuzleri hizlica tespit et
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in faces:
                # Saci da kapsayacak sekilde yuzu biraz genislet (padding)
                pad_y_top = int(h * 0.4)  # Yukari (sac)
                pad_y_bot = int(h * 0.2)  # Asagi (boyun/cene)
                pad_x = int(w * 0.2)      # Yanlar
                
                y1 = max(0, y - pad_y_top)
                y2 = min(frame.shape[0], y + h + pad_y_bot)
                x1 = max(0, x - pad_x)
                x2 = min(frame.shape[1], x + w + pad_x)
                
                # Bu bolgeyi siyaha boyayarak (0), mont olarak algilanmasini TAMAMEN ENGELLERIZ
                # Boyleyce yuz ve sac her zaman 'canli kare' olarak kalir
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        mask_inv = cv2.bitwise_not(mask)

        bg = self.cloak_bg
        if bg.shape != frame.shape:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
            self.cloak_bg = bg

        # Arka plani STATIK hafizadan aliyoruz (sadece mont siyahligi icin)
        # Gercek canli kareyi kullaniyoruz (böylece sizin yuzunuz, kollarınız canlı kalıyor)
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        bg_part = cv2.bitwise_and(self.cloak_bg, self.cloak_bg, mask=mask)
        result = cv2.add(fg, bg_part)

        return result

    # ================================================================
    # Dosya
    # ================================================================
    def load_image(self):
        if self.camera_active:
            self._stop_camera()
        self._set_mode(MODE_NORMAL)

        path = filedialog.askopenfilename(
            title="Bir resim secin",
            filetypes=[("Goruntu Dosyalari", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Hata", "Goruntu okunamadi!")
            return

        self.image_path = path
        self.original_img = img.copy()
        self.working_img = img.copy()
        self.backup_img = None
        self.erase_mask = None
        h, w = img.shape[:2]
        self._status(f"{os.path.basename(path)}  |  {w}x{h} piksel  |  Yuklendi")
        self.update_image()

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("Uyari", "Kaydedilecek goruntu yok!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", title="Islenmis Resmi Kaydet",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Tum Dosyalar", "*.*")]
        )
        if path:
            cv2.imwrite(path, self.processed_img)
            self._status(f"Kaydedildi: {os.path.basename(path)}")

    # ================================================================
    # Sifirlama & Geri Alma
    # ================================================================
    def _save_backup(self):
        if self.working_img is not None:
            self.backup_img = self.working_img.copy()

    def _undo(self):
        if self.backup_img is not None:
            self.working_img = self.backup_img.copy()
            self.backup_img = None
            self.erase_mask = None
            self.update_image()
            self._status("Son islem geri alindi")
        else:
            messagebox.showinfo("Bilgi", "Geri alinacak islem yok!")

    def reset_all(self):
        self._set_mode(MODE_NORMAL)
        self._clear_crop_rect()
        self.scale_samp.set(1)
        self.scale_quant.set(8)
        self.scale_ksize.set(3)
        self.filter_var.set("Yok")
        self.erase_mask = None
        if self.original_img is not None:
            self.working_img = self.original_img.copy()
            self.backup_img = None
            self.update_image()
            self._status("Tum efektler ve duzenlemeler sifirlandi")
        else:
            self._status("Degerler sifirlandi")

    def clear_image(self):
        if self.camera_active:
            self._stop_camera()
        self._set_mode(MODE_NORMAL)
        self._clear_crop_rect()
        self.original_img = None
        self.working_img = None
        self.processed_img = None
        self.backup_img = None
        self.erase_mask = None
        self.image_path = None
        self._show_placeholder()
        self._clear_histogram()
        self._status("Resim kaldirildi")

    # ================================================================
    # Goruntu Isleme Pipeline
    # ================================================================
    def update_image(self):
        if self.working_img is None:
            return

        img = self.working_img.copy()
        effects = []

        # 1 - Ornekleme
        samp = self.scale_samp.get()
        if samp > 1:
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(w // samp, 1), max(h // samp, 1)),
                               interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            effects.append(f"Ornekleme x{samp}")

        # 2 - Kuantalama
        bits = self.scale_quant.get()
        if bits < 8:
            factor = 256 // (2 ** bits)
            img = ((img // factor) * factor).astype(np.uint8)
            effects.append(f"Kuantalama {bits}-bit")

        # 3 - Filtre
        f_type = self.filter_var.get()
        k_size = self.scale_ksize.get()
        if k_size % 2 == 0:
            k_size += 1
        if f_type != "Yok":
            img = self._apply_filter(img, f_type, k_size)
            effects.append(f"{f_type} k={k_size}")

        self.processed_img = img

        if not self.camera_active and effects:
            self._status("  |  ".join(effects))

        self._display(img)
        self._update_histogram(img)

    def _apply_filter(self, img, f_type, k):
        if f_type == "Gaussian Blur":
            return cv2.GaussianBlur(img, (k, k), 0)
        if f_type == "Median Blur":
            return cv2.medianBlur(img, k)
        if f_type == "Dilation (Genisletme)":
            return cv2.dilate(img, np.ones((k, k), np.uint8), iterations=1)
        if f_type == "Erosion (Asindirma)":
            return cv2.erode(img, np.ones((k, k), np.uint8), iterations=1)
        if f_type == "Sobel (Kenar Bulma)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ks = min(k, 31)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ks)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ks)
            mag = cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        if f_type == "Canny (Kenar Bulma)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kc = min(max(k, 3), 7)
            if kc % 2 == 0:
                kc -= 1
            v = float(np.median(gray))
            lo, hi = int(max(0, 0.67 * v)), int(min(255, 1.33 * v))
            edges = cv2.Canny(gray, lo, hi, apertureSize=kc)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return img

    # ================================================================
    # Goruntuleme
    # ================================================================
    def _display(self, img_bgr):
        self.img_canvas.delete("all")
        self._canvas_img_id = None

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        cw = max(self.img_canvas.winfo_width(), 200)
        ch = max(self.img_canvas.winfo_height(), 200)

        pil_img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        dw, dh = pil_img.size
        ih, iw = img_bgr.shape[:2]

        self._disp_scale = dw / iw if iw > 0 else 1.0
        self._disp_w = dw
        self._disp_h = dh
        self._disp_offset_x = (cw - dw) // 2
        self._disp_offset_y = (ch - dh) // 2

        tk_img = ImageTk.PhotoImage(pil_img)
        self._canvas_img_id = self.img_canvas.create_image(
            self._disp_offset_x, self._disp_offset_y,
            anchor="nw", image=tk_img
        )
        self.img_canvas._tk_img = tk_img

    def _status(self, text):
        self.status_bar.configure(text="  " + text)

    def _on_close(self):
        if self.camera_active:
            self._stop_camera()
        self.root.destroy()


# ================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoshopApp(root)
    root.mainloop()
