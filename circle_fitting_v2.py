import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from skimage.measure import CircleModel, ransac


#==================================
# ===== PARAMETER =====
#==================================
DISPLAY_SCALE = 0.4 # Ausgabegröße der Bilder   
IMAGE_PATH = "Test.png" # Dateiname   
GRAY_VALUE = 70 # Threshhold für Binärisierung   
OFFSET_MM = 6 # Breite der Probe   
X_OR_Y = 1 # 1 Y Abstand = X Abstand   

#==================================
# ===== FUNKTIONSDEFINITIONEN =====
#==================================
reference_points = []


def ask_reference_length_mm():
    while True:
        user_input = input("Referenzlaenge in mm eingeben: ").strip().replace(",", ".")
        try:
            val = float(user_input)
            if val > 0:
                return val
        except:
            pass
        print("Ungueltige Eingabe, bitte erneut.")

def select_reference_points(image):
    global reference_points

    clone_full = image.copy()
    h, w = clone_full.shape[:2]

    # Zoom-Parameter
    zoom = 1.0
    zoom_min = DISPLAY_SCALE
    zoom_max = 5.0

    view_center = np.array([w / 2, h / 2], dtype=np.float64)

    def get_view():
        if zoom <= 1.0:
            return cv2.resize(clone_full, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        view_w = int(w / zoom)
        view_h = int(h / zoom)

        cx, cy = view_center
        x1 = int(np.clip(cx - view_w // 2, 0, w - view_w))
        y1 = int(np.clip(cy - view_h // 2, 0, h - view_h))

        crop = clone_full[y1:y1+view_h, x1:x1+view_w]
        return cv2.resize(crop, None, fx=DISPLAY_SCALE * zoom, fy=DISPLAY_SCALE * zoom)

    def screen_to_image(x, y):
        if zoom <= 1.0:
            return int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE)

        view_w = int(w / zoom)
        view_h = int(h / zoom)

        cx, cy = view_center
        x1 = int(np.clip(cx - view_w // 2, 0, w - view_w))
        y1 = int(np.clip(cy - view_h // 2, 0, h - view_h))

        scale = DISPLAY_SCALE * zoom

        x_img = x1 + int(x / scale)
        y_img = y1 + int(y / scale)

        return np.clip(x_img, 0, w-1), np.clip(y_img, 0, h-1)

    def mouse(event, x, y, flags, param):
        nonlocal zoom, view_center

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000

            if delta > 0:
                zoom = min(zoom * 1.2, zoom_max)
            else:
                zoom = max(zoom / 1.2, zoom_min)

        elif event == cv2.EVENT_LBUTTONDOWN and len(reference_points) < 2:
            x_img, y_img = screen_to_image(x, y)
            reference_points.append((x_img, y_img))

    cv2.namedWindow("Referenz setzen (Zoom + ENTER)")
    cv2.setMouseCallback("Referenz setzen (Zoom + ENTER)", mouse)

    while True:
        display = get_view()

        def image_to_screen(p):
            if zoom <= 1.0:
                return int(p[0] * DISPLAY_SCALE), int(p[1] * DISPLAY_SCALE)

            view_w = int(w / zoom)
            view_h = int(h / zoom)

            cx, cy = view_center
            x1 = int(np.clip(cx - view_w // 2, 0, w - view_w))
            y1 = int(np.clip(cy - view_h // 2, 0, h - view_h))

            scale = DISPLAY_SCALE * zoom

            x_screen = int((p[0] - x1) * scale)
            y_screen = int((p[1] - y1) * scale)

            return x_screen, y_screen

        # Punkte korrekt zeichnen
        for p in reference_points:
            px, py = image_to_screen(p)
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)

        cv2.imshow("Referenz setzen (Zoom + ENTER)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(reference_points) == 2:
            break

        # optional: WASD panning
        if key == ord('w'):
            view_center[1] -= 50 / zoom
        elif key == ord('s'):
            view_center[1] += 50 / zoom
        elif key == ord('a'):
            view_center[0] -= 50 / zoom
        elif key == ord('d'):
            view_center[0] += 50 / zoom

        view_center[0] = np.clip(view_center[0], 0, w)
        view_center[1] = np.clip(view_center[1], 0, h)

    cv2.destroyAllWindows()

    p1 = np.array(reference_points[0])
    p2 = np.array(reference_points[1])

    return abs(p2[X_OR_Y] - p1[X_OR_Y])

# Globale Variable für den aktuellen Lasso-Pfad
current_lasso_path = None

def on_select(vertices):
    """ Wird aufgerufen, sobald eine Lasso-Form geschlossen wird """
    global current_lasso_path
    current_lasso_path = Path(vertices)
    plt.close()

def get_lasso_selection(display_image, window_title):
    """ Öffnet das Lasso-Fenster auf dem übergebenen Bild (z.B. Canny) """
    global current_lasso_path
    current_lasso_path = None
    
    fig, ax = plt.subplots()
    
    if len(display_image.shape) == 2:
        ax.imshow(display_image, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        
    plt.title(window_title)
    
    selector = LassoSelector(ax, on_select, props=dict(color='green', linewidth=2))
    plt.show()
    
    return current_lasso_path


#==================================
#   ===== HAUPTPROGRAMM =====
#==================================

def main():
    global reference_points

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fehler beim Laden des Bildes.")
        return

    # ===== REFERENZ =====
    reference_points = []  # wichtig: reset
    reference_length_px = select_reference_points(img)
    reference_length_mm = ask_reference_length_mm()

    mm_per_px = reference_length_mm / reference_length_px
    print(f"Skalierung (manuell): {mm_per_px:.6f} mm/px")

    # ===== ROI =====
    print("ROI auswählen")

    small_img = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    roi_small = cv2.selectROI(
        "ROI Auswahl",
        small_img,
        fromCenter=False
    )
    cv2.destroyAllWindows()

    x_s, y_s, w_s, h_s = roi_small

    x = int(x_s / DISPLAY_SCALE)
    y = int(y_s / DISPLAY_SCALE)
    w = int(w_s / DISPLAY_SCALE)
    h = int(h_s / DISPLAY_SCALE)

    if (w > 0) and (h > 0):
        roi = img[y:y+h, x:x+w]
        roi_offset = (x, y)
    else:
        roi = img
        roi_offset = (0, 0)

    # ===== PREPROCESS =====
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, GRAY_VALUE, 255, cv2.THRESH_BINARY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)

    edges = cv2.Canny(thresh, 50, 150)

    # ---------- COLLAGE DEBUG ----------
    scale = 4 * DISPLAY_SCALE  # <--- hier Größe einstellen

    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    gray_eq_color = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    roi_small = cv2.resize(roi, None, fx=scale, fy=scale)
    thresh_small = cv2.resize(thresh_color, None, fx=scale, fy=scale)
    edges_small = cv2.resize(edges_color, None, fx=scale, fy=scale)
    gray_eq_small = cv2.resize(gray_eq_color, None, fx=scale, fy=scale)

    def label(img, text):
        cv2.putText(img, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        return img

    roi_small = label(roi_small, "ROI")
    thresh_small = label(thresh_small, "Threshold")
    edges_small = label(edges_small, "Canny")
    gray_eq_small = label(gray_eq_small, "HistEq")

    collage = np.vstack((np.hstack((roi_small, thresh_small)), np.hstack((edges_small, gray_eq_small))))

    cv2.imshow("Debug Collage", collage)
    cv2.waitKey(1)

    edge_coords = cv2.findNonZero(edges)
    if edge_coords is None:
        print("Keine Kanten im ROI-Bereich gefunden.")
        return

    edge_points = np.squeeze(edge_coords)

    # ===== LASSO =====
    path1 = get_lasso_selection(edges, "KREIS 1")

    if path1 is None:
        print("Abbruch.")
        return

    inside_mask1 = path1.contains_points(edge_points)
    filtered_points1 = edge_points[inside_mask1]

# ===== OPTIMIERTES ROBUSTES FITTING (RANSAC) =====
    model_robust, inliers = ransac(
        filtered_points1,
        CircleModel,
        min_samples=3,
        residual_threshold=1.0,  # Maximale Abweichung in Pixeln für Inlier
        max_trials=1500
    )
    xc_roi1, yc_roi1, r1 = model_robust.params
    print(f"RANSAC erfolgreich: {np.sum(inliers)} von {len(filtered_points1)} Punkten genutzt.")

    # ---------------------------------------------------------
    # NEU: VISUALISIERUNG DER VERWORFENEN PIXEL (DEBUG)
    # ---------------------------------------------------------
    # Trennung der Punkte in Inlier (gut) und Outlier (verworfen)
    inlier_points = filtered_points1[inliers]
    outlier_points = filtered_points1[~inliers]

    # Wir erstellen ein farbiges Debug-Bild basierend auf dem Canny-Kantenbild
    ransac_debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 1. Genutzte Punkte (Inlier) GRÜN einzeichnen
    for pt in inlier_points:
        cv2.circle(ransac_debug, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

    # 2. Verworfene Punkte (Outlier/Aussparungen) ROT und etwas größer einzeichnen
    for pt in outlier_points:
        cv2.circle(ransac_debug, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

    # Bild für die Anzeige skalieren und anzeigen
    ransac_debug_small = cv2.resize(ransac_debug, None, fx=5*DISPLAY_SCALE, fy=5*DISPLAY_SCALE)
    cv2.imshow("RANSAC Debug: Inlier (Gruen) vs Outlier (Rot)", ransac_debug_small)
    cv2.waitKey(1)  # Wechselt automatisch weiter, sobald das Lasso schließt
    # ---------------------------------------------------------

    xc_abs1 = xc_roi1 + roi_offset[0]
    yc_abs1 = yc_roi1 + roi_offset[1]

    r1_mm = r1 * mm_per_px
    r2 = r1 - (OFFSET_MM / mm_per_px)

    # ===== ERGEBNIS =====
    result_img = img.copy()

    cv2.circle(result_img, (int(xc_abs1), int(yc_abs1)), int(r1), (255, 255, 0), 2)
    cv2.circle(result_img, (int(xc_abs1), int(yc_abs1)), 4, (0, 0, 255), -1)
    cv2.circle(result_img, (int(xc_abs1), int(yc_abs1)), int(r2), (0, 255, 255), 2)

    cv2.putText(result_img, f"Kreis1: {r1:.2f}px | {r1_mm:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, f"Kreis2: {r2:.2f}px | {r2 * mm_per_px:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    print(f"r1 = {r1_mm}")
    print(f"r2 = {r2*mm_per_px}")
    
    # ===== SPEICHERN =====
    print("\n" + "="*50)
    filename_input = input("Dateiname: ").strip()

    if not filename_input:
        filename_input = "messung_standard"

    if not filename_input.lower().endswith(('.jpg', '.jpeg', '.png')):
        filename_input += ".jpg"

    cv2.imwrite(filename_input, result_img)

    print(f"✓ gespeichert: {filename_input}")
    print("="*50 + "\n")

    Ergebnis_Kopie = result_img.copy()
    Ergebnis_Kopie = cv2.resize(Ergebnis_Kopie, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Ergebnis", Ergebnis_Kopie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
  
