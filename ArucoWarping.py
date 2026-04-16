import cv2
import numpy as np

# --- 0. Konfiguration & Parameter ---
# Skalierungsfaktor: 1mm im echten Leben = 5 Pixel im Bild
# Erhöhe diesen Wert für mehr Details, verringere ihn für kleinere Dateien
PX_PER_MM = 5 

# Die echte Breite eines Markers in Millimetern (bitte exakt nachmessen!)
MARKER_SIZE_MM = 40  

# Dein händisch gemessenes Layout in mm
MARKER_LAYOUT_MM = {
    0:  (0,   0),
    5:  (174, 0),
    12: (0,   61),
    23: (174, 61),
    42: (0,   122),
    87: (174, 122),
}

# --- 1. Daten laden (wie gehabt) ---
# (Ich gehe davon aus, dass matrix, dist_coeffs und image bereits geladen sind)
# undistorted = cv2.undistort(image, matrix, dist_coeffs)

# --- 2. Marker Detektion ---
marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(undistorted)

if marker_ids is not None and len(marker_ids) >= 2:
    src_pts = []
    dst_pts = []

    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        if marker_id not in MARKER_LAYOUT_MM:
            continue

        # Bildpunkte aus der Kamera (4 Ecken des aktuellen Markers)
        img_corners = corners.reshape(4, 2).astype(np.float32)

        # Ziel-Koordinaten in mm aus dem Layout holen
        x_mm, y_mm = MARKER_LAYOUT_MM[marker_id]

        # Die 4 Ecken im Zielbild definieren (in mm, dann skaliert auf Pixel)
        # Wichtig: Reihenfolge muss ArUco Standard entsprechen (TL, TR, BR, BL)
        target_corners_mm = np.array([
            [x_mm, y_mm],                                 # Oben Links
            [x_mm + MARKER_SIZE_MM, y_mm],                # Oben Rechts
            [x_mm + MARKER_SIZE_MM, y_mm + MARKER_SIZE_MM], # Unten Rechts
            [x_mm, y_mm + MARKER_SIZE_MM],                # Unten Links
        ], dtype=np.float32)

        # Umrechnung in Ziel-Pixel
        dst_corners = target_corners_mm * PX_PER_MM

        src_pts.append(img_corners)
        dst_pts.append(dst_corners)

    if len(src_pts) >= 2:
        src_pts = np.vstack(src_pts)
        dst_pts = np.vstack(dst_pts)

        # Homographie berechnen
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Dynamische Berechnung der Ausgabegröße basierend auf den mm-Angaben
        max_x_mm = max([pos[0] for pos in MARKER_LAYOUT_MM.values()]) + MARKER_SIZE_MM
        max_y_mm = max([pos[1] for pos in MARKER_LAYOUT_MM.values()]) + MARKER_SIZE_MM
        
        out_w = int(max_x_mm * PX_PER_MM)
        out_h = int(max_y_mm * PX_PER_MM)

        # Warping anwenden
        rectified = cv2.warpPerspective(undistorted, H, (out_w, out_h))

        # Ergebnisse anzeigen
        cv2.imshow("Rectified Top-View", rectified)
        cv2.waitKey(0)
    else:
        print("Nicht genügend bekannte Marker im Bild gefunden.")
else:
    print("Keine Marker erkannt.")
