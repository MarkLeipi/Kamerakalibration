############################################################
# Kantendetektion + Kreisfitting + Formanalyse
# REIN INTERAKTIVE px → mm SKALIERUNG (pan/zoom-unabhängig)
############################################################

import cv2
import numpy as np
from set_params import RECTIFIED_TOP_VIEW_PATH


# ==========================================================
# Bild laden
# ==========================================================

img = cv2.imread(RECTIFIED_TOP_VIEW_PATH)
if img is None:
    raise IOError(f"Bild konnte nicht geladen werden: {RECTIFIED_TOP_VIEW_PATH}")


# ==========================================================
# Kantendetektion → Kreisfilter
# ==========================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, threshold1=60, threshold2=150, L2gradient=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

def is_circle(cnt, min_area=200, max_area=20000, min_circ=0.80, min_solid=0.95):
    area = cv2.contourArea(cnt)
    if not (min_area < area < max_area):
        return False
    
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter**2)
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    
    hull = cv2.convexHull(cnt)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    
    return (circularity > min_circ and 
            0.85 < aspect_ratio < 1.15 and 
            solidity > min_solid)

circle_contours = [cnt for cnt in contours if is_circle(cnt)]
print(f"Erkannte Kreise: {len(circle_contours)}")


# ==========================================================
# Kreisfitting + RMS
# ==========================================================

def fit_circle(points):
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    A = np.column_stack([2*x, 2*y, np.ones(len(points))])
    b = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = C[0], C[1]
    r = np.sqrt(C[2] + cx**2 + cy**2)
    return cx, cy, r


def circle_rms(points, cx, cy, r):
    d = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    return np.sqrt(np.mean((d - r)**2))


results = []
for cnt in circle_contours:
    pts = cnt.reshape(-1, 2)
    cx, cy, r = fit_circle(pts)
    if r >= 20.0:
        rms = circle_rms(pts, cx, cy, r)
        results.append((cx, cy, r, rms))

if results:
    radii = np.array([r for _, _, r, _ in results])
    rms_errors = np.array([rms for _, _, _, rms in results])
    rms_mean, rms_std = np.mean(rms_errors), np.std(rms_errors)
else:
    rms_mean = rms_std = 1.0


# ==========================================================
# Basis-Visualisierung
# ==========================================================

vis = img.copy()
for cx, cy, r, rms in results:
    cx_i, cy_i, r_i = int(cx), int(cy), int(r)
    
    z = min(abs(rms - rms_mean) / (rms_std + 1e-6), 2.0)
    color = (int(255 * z / 2), int(255 * (1 - z / 2)), 0)
    
    cv2.circle(vis, (cx_i, cy_i), r_i, color, 2)
    cv2.circle(vis, (cx_i, cy_i), 2, (0, 0, 255), -1)


# ==========================================================
# Interaktive Messung (nur Klicken für Referenzlänge)
# ==========================================================

class InteractiveMeasure:
    def __init__(self, image):
        self.base = image
        self.zoom = 1.0
        self.offset = np.array([50, 50], dtype=float)
        
        self.ref_pt1 = None
        self.ref_pt2 = None
        self.measure_pts = []
        self.mm_per_px = None
        
        self.dragging = False
        self.mode = "reference"  # "reference" oder "measure"

    def screen_to_img(self, x, y):
        return (np.array([x, y], dtype=float) - self.offset) / self.zoom

    def img_to_screen(self, p):
        return (p * self.zoom + self.offset).astype(int)

    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = self.screen_to_img(x, y)
            if self.mode == "reference":
                self.ref_pt1 = pt if self.ref_pt1 is None else self.ref_pt2
                self.ref_pt2 = pt
            else:  # measure
                self.measure_pts.append(pt)
                if len(self.measure_pts) > 2:
                    self.measure_pts.pop(0)

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.last_mouse = np.array([x, y])

        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            delta = np.array([x, y]) - self.last_mouse
            self.offset += delta
            self.last_mouse = np.array([x, y])

        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom *= 1.1 if flags > 0 else 0.9
            self.zoom = np.clip(self.zoom, 0.2, 10.0)

    def render(self):
        h, w = self.base.shape[:2]
        view = cv2.resize(self.base, (int(w*self.zoom), int(h*self.zoom)))
        canvas = np.zeros((900, 1400, 3), dtype=np.uint8) + 30
        
        ox, oy = self.offset.astype(int)
        x1, y1 = max(0, ox), max(0, oy)
        x2, y2 = min(canvas.shape[1], ox + view.shape[1]), min(canvas.shape[0], oy + view.shape[0])
        vx1, vy1 = max(0, -ox), max(0, -oy)
        vx2, vy2 = vx1 + (x2 - x1), vy1 + (y2 - y1)
        
        canvas[y1:y2, x1:x2] = view[vy1:vy2, vx1:vx2]

        # Referenzmessung
        if self.ref_pt1 is not None and self.ref_pt2 is not None:
            p1, p2 = self.img_to_screen(self.ref_pt1), self.img_to_screen(self.ref_pt2)
            cv2.line(canvas, tuple(p1), tuple(p2), (255, 0, 0), 2)
            cv2.circle(canvas, tuple(p1), 5, (255, 0, 0), -1)
            cv2.circle(canvas, tuple(p2), 5, (255, 0, 0), -1)
            
            px = np.linalg.norm(self.ref_pt2 - self.ref_pt1)
            if self.mm_per_px:
                txt = f"Referenz: {px * self.mm_per_px:.2f} mm ({px:.1f} px)"
                cv2.putText(canvas, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Messung
        if len(self.measure_pts) == 2:
            p1, p2 = self.img_to_screen(self.measure_pts[0]), self.img_to_screen(self.measure_pts[1])
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 255, 255), 2)
            cv2.circle(canvas, tuple(p1), 5, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(p2), 5, (0, 0, 255), -1)
            
            px = np.linalg.norm(self.measure_pts[1] - self.measure_pts[0])
            txt = f"Messung: {px:.1f} px"
            if self.mm_per_px:
                txt += f" = {px * self.mm_per_px:.2f} mm"
            cv2.putText(canvas, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Status
        status = "Modus: REFERENZ eingeben" if self.mode == "reference" else "Modus: MESSUNG"
        color = (0, 0, 255) if self.mode == "reference" else (0, 255, 0)
        cv2.putText(canvas, status, (20, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if self.mm_per_px:
            cv2.putText(canvas, f"Skalierung: {self.mm_per_px:.6f} mm/px",
                       (20, 880), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return canvas


# ==========================================================
# App starten
# ==========================================================

viewer = InteractiveMeasure(vis)

cv2.namedWindow("Messung", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Messung", viewer.mouse)

print("\n" + "="*50)
print("INTERAKTIVE LÄNGENMESSUNG")
print("="*50)
print("1. Linksklick: Referenzlänge setzen (2 Punkte)")
print("2. Taste 'e':  Referenzlänge eingeben (mm)")
print("3. Taste 'm':  Zum Messmodus wechseln")
print("4. Mittl. T.:  Pan")
print("5. Mausrad:    Zoom")
print("6. Taste 'r':  Reset")
print("7. ESC:        Ende\n")

while True:
    frame = viewer.render()
    cv2.imshow("Messung", frame)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

    if key == ord('e') and viewer.ref_pt1 is not None and viewer.ref_pt2 is not None:
        try:
            mm_ref = float(input("Referenzlänge [mm]: "))
            px_ref = np.linalg.norm(viewer.ref_pt2 - viewer.ref_pt1)
            viewer.mm_per_px = mm_ref / px_ref
            viewer.mode = "measure"
            print(f"✓ Skalierung gesetzt: {viewer.mm_per_px:.6f} mm/px")
        except ValueError:
            print("✗ Ungültige Eingabe!")

    if key == ord('m'):
        viewer.mode = "measure" if viewer.mode == "reference" else "reference"

    if key == ord('r'):
        viewer.ref_pt1 = viewer.ref_pt2 = viewer.measure_pts = []
        viewer.mm_per_px = None
        viewer.mode = "reference"

cv2.destroyAllWindows()

