"""
Funktionalität:
1. Kalibrierung durchführen
2. RMS-Fehler pro Bild ausgeben
3. Gesamt-RMS ausgeben  
4. Optional: Debug-Bilder speichern
"""

from pathlib import Path
import cv2
import numpy as np

from opencv_support import create_charuco_runtime
from set_params import (
    CAL_IMAGES_DIR,
    CALIBRATION_DEBUG_OUTPUT_DIR,  
    CAMERA_MATRIX_PATH,
    DIST_COEFFS_PATH,
    SAVE_DEBUG,
    NMB_DEBUG,
    create_charuco_board,
    create_charuco_dictionary,
    ensure_output_directories,
)

def calibrate():
    """Führt Kalibrierung durch mit Fokus auf RMS-Ausgabe und Debugging."""
    board = create_charuco_board()
    runtime = create_charuco_runtime(board, create_charuco_dictionary())
    ensure_output_directories()
    
    # Bilder laden
    image_paths = sorted([p for p in CAL_IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Verwende: {runtime.description}\nBilder gefunden: {len(image_paths)}\n")
    
    all_object_points = []
    all_image_points = []
    all_images = []
    image_size = None

    # 1. Detektion & Punkt-Matching
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None: continue
        if image_size is None: image_size = (image.shape[1], image.shape[0])

        # Marker und ChArUco-Ecken erkennen (wie im alten Code)
        marker_corners, marker_ids, _ = runtime.detect_markers(image)
        charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(image, marker_corners, marker_ids)

        if charuco_count >= 4:
            # Verbindung zu 3D-Board-Punkten (matchImagePoints wie gewünscht)
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            
            if obj_pts is not None and img_pts is not None:
                all_object_points.append(obj_pts)
                all_image_points.append(img_pts)
                all_images.append(image)
                print(f"✓ {path.name}: {len(obj_pts)} Punkte erkannt")
                
                # Debug-Bild (Erkennung)
                if SAVE_DEBUG and len(all_images) <= NMB_DEBUG:
                    debug_img = image.copy()
                    cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
                    runtime.draw_charuco_corners(debug_img, charuco_corners, charuco_ids)
                    cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{len(all_images)}.jpg"), debug_img)
        else:
            print(f"✗ {path.name}: Zu wenig Punkte")

    if len(all_object_points) < 10:
        raise RuntimeError(f"Nur {len(all_object_points)} Bilder brauchbar - brauche 10!")

    # 2. Kalibrierung
    # Wir nutzen hier cv2.calibrateCamera, da wir die Punkte bereits via matchImagePoints 
    # in das passende Format (Objekt- vs Bildpunkte) gebracht haben.
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None
    )

    # 3. Ausgaben (Logs & RMS)
    print(f"\n{len(all_object_points)}/{len(image_paths)} Bilder verwendbar")
    print(f"Gesamt-RMS: {retval:.6f}\n")
    print("RMS-Fehler pro Bild:")

    for i, (obj_pts, img_pts, img) in enumerate(zip(all_object_points, all_image_points, all_images)):
        # Reprojektion
        projected, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_pts, projected, cv2.NORM_L2)
        rms = np.sqrt((error ** 2) / len(projected))
        print(f"  Bild {i+1}: {rms:.6f}")

        # Optional: Undistorted Debug-Bild speichern
        if SAVE_DEBUG and i < NMB_DEBUG:
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
            cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i+1}_undistorted.jpg"), undistorted)

    # 4. Ergebnisse speichern
    np.save(str(CAMERA_MATRIX_PATH), camera_matrix)
    np.save(str(DIST_COEFFS_PATH), dist_coeffs)
    print(f"\nErgebnisse gespeichert in: {CALIBRATION_DEBUG_OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        calibrate()
    finally:
        cv2.destroyAllWindows()
