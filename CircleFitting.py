############################################################
# Kantendetektion + Kreisfitting + Formanalyse
# REIN INTERAKTIVE px → mm SKALIERUNG (pan/zoom-unabhängig)
############################################################

import cv2
import numpy as np
from set_params import RECTIFIED_TOP_VIEW_PATH


# ==========================================================
# 1. Bild laden
# ==========================================================

IMAGE_PATH = RECTIFIED_TOP_VIEW_PATH
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise IOError(f"Bild konnte nicht geladen werden: {IMAGE_PATH}")

MIN_RADIUS_PX = 20.0   # minimaler Radius


# ==========================================================
# 2. Kantendetektion
# ==========================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(
    blur,
    threshold1=60,
    threshold2=150,
    L2gradient=True
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# ==========================================================
# 3. Konturen → Kreisfilter
# ==========================================================

contours, _ = cv2.findContours(
    edges_closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

circle_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200 or area > 20000:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter**2)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h

    hull = cv2.convexHull(cnt)
    solidity = area / cv2.contourArea(hull)

    if (
        circularity > 0.80 and
        0.85 < aspect_ratio < 1.15 and
        solidity > 0.95
    ):
        circle_contours.append(cnt)

print(f"Erkannte Kreise: {len(circle_contours)}")


# ==========================================================
# 4. Kreisfitting + RMS
# ==========================================================

def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]

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
radii = []
rms_errors = []

for cnt in circle_contours:
    pts = cnt.reshape(-1, 2)
    cx, cy, r = fit_circle(pts)

    if r < MIN_RADIUS_PX:
        continue

    rms = circle_rms(pts, cx, cy, r)
    results.append((cx, cy, r, rms))
    radii.append(r)
    rms_errors.append(rms)


# ==========================================================
# 5. Statistik (noch ohne mm!)
# ==========================================================

r_mean = np.mean(radii)
r_std  = np.std(radii)
rms_mean = np.mean(rms_errors)
rms_std  = np.std(rms_errors)


# ==========================================================
# 6. Visualisierung Kreise
# ==========================================================

vis = img.copy()

for cx, cy, r, rms in results:
    cx_i, cy_i = int(cx), int(cy)
    r_i = int(r)

    z = abs(rms - rms_mean) / (rms_std + 1e-6)
    z = min(z, 2.0)

    color = (
        int(255 * z / 2),
        int(255 * (1 - z / 2)),
        0
    )

    cv2.circle(vis, (cx_i, cy_i), r_i, color, 2)
    cv2.circle(vis, (cx_i, cy_i), 2, (0, 0, 255), -1)


# ==========================================================
# 7. Interaktive Messung (rein bildkoordinatenbasiert)
# ==========================================================

class InteractiveMeasure:
    def __init__(self, image):
        self.base = image
        self.zoom = 1.0
        self.offset = np.array([50, 50], dtype=float)

        self.ref_pts = []
        self.mm_per_px = None

        self.dragging = False
        self.last_mouse = None

    # --- Koordinaten ---
    def screen_to_img(self, x, y):
        return (np.array([x, y]) - self.offset) / self.zoom

    def img_to_screen(self, p):
        return (p * self.zoom + self.offset).astype(int)

    # --- Maus ---
    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = self.screen_to_img(x, y)
            self.ref_pts.append(pt)
            if len(self.ref_pts) > 2:
                self.ref_pts.pop(0)

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

    # --- Render ---
    def render(self):
        h, w = self.base.shape[:2]
        view = cv2.resize(self.base, (int(w*self.zoom), int(h*self.zoom)))

        canvas = np.zeros((900, 1400, 3), dtype=np.uint8) + 30

        ox, oy = self.offset.astype(int)

        x1 = max(0, ox)
        y1 = max(0, oy)
        x2 = min(canvas.shape[1], ox + view.shape[1])
        y2 = min(canvas.shape[0], oy + view.shape[0])

        vx1 = max(0, -ox)
        vy1 = max(0, -oy)
        vx2 = vx1 + (x2 - x1)
        vy2 = vy1 + (y2 - y1)

        canvas[y1:y2, x1:x2] = view[vy1:vy2, vx1:vx2]

        if len(self.ref_pts) == 2:
            p1 = self.img_to_screen(self.ref_pts[0])
            p2 = self.img_to_screen(self.ref_pts[1])

            cv2.line(canvas, tuple(p1), tuple(p2), (0,255,255), 2)
            cv2.circle(canvas, tuple(p1), 5, (0,0,255), -1)
            cv2.circle(canvas, tuple(p2), 5, (0,0,255), -1)

            px = np.linalg.norm(self.ref_pts[1] - self.ref_pts[0])
            txt = f"{px:.2f} px"
            if self.mm_per_px:
                txt += f" = {px * self.mm_per_px:.2f} mm"

            cv2.putText(canvas, txt, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if self.mm_per_px:
            cv2.putText(
                canvas,
                f"Skalierung: {self.mm_per_px:.6f} mm/px",
                (20,75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0), 2
            )

        return canvas


# ==========================================================
# 8. App starten
# ==========================================================

viewer = InteractiveMeasure(vis)

cv2.namedWindow("Messung", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Messung", viewer.mouse)

print("\nINFO:")
print("  Linksklick     → 2 Referenzpunkte")
print("  Mittlere Taste → Pan")
print("  Mausrad        → Zoom")
print("  Taste 'c'      → Skalierung eingeben")
print("  Taste 'r'      → Reset")
print("  ESC            → Ende\n")

while True:
    frame = viewer.render()
    cv2.imshow("Messung", frame)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

    if key == ord('r'):
        viewer.ref_pts = []
        viewer.mm_per_px = None

    if key == ord('c') and len(viewer.ref_pts) == 2:
        px = np.linalg.norm(viewer.ref_pts[1] - viewer.ref_pts[0])
        mm = float(input("Referenzlänge in mm: "))
        viewer.mm_per_px = mm / px
        print(f"Skalierung gesetzt: {viewer.mm_per_px:.6f} mm/px")

cv2.destroyAllWindows()
