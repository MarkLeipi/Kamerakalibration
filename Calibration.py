"""
Vereinfachte Kamerakalibrierung mit ChArUco-Board.

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
    CALIBRATION_DEBUG_OUTPUT_DIR,  # ===== ZUM ENTFERNEN ===== Nur wenn save_debug=False dauerhaft gesetzt wird (Zeile 88)
    CAMERA_MATRIX_PATH,
    DIST_COEFFS_PATH,
    create_charuco_board,
    create_charuco_dictionary,
    ensure_output_directories,
)


def load_all_images():
    """Laden aller Kalibierbilder."""
    if not CAL_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {CAL_IMAGES_DIR}")
    
    supported = {".jpg", ".jpeg", ".png"}
    images = sorted(
        path for path in CAL_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )
    return images


def detect_corners_in_image(image, board, runtime, image_path, save_debug, debug_counter):
    """
    Erkennt ChArUco-Ecken in einem Bild.
    
    Rückgabe: (object_points, image_points, erfolgreich) oder (None, None, False)
    """
    # Marker erkennen
    marker_corners, marker_ids, _ = runtime.detect_markers(image)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, False
    
    # ChArUco-Ecken aus Markern extrahieren
    charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(
        image, marker_corners, marker_ids
    )
    
    # Mindestens 4 Ecken nötig
    if charuco_count < 4 or charuco_corners is None or charuco_ids is None:
        return None, None, False
    
    # Bild-Punkte mit 3D Board-Punkte verbinden
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    
    if object_points is None or image_points is None or len(object_points) < 4:
        return None, None, False
    
    # ===== ZUM ENTFERNEN ===== Dieser ganze if-Block kann gelöscht werden, wenn save_debug nicht mehr nötig ist
    # Optional: Debug-Bild speichern (nur erste 3 Bilder)
    if save_debug and debug_counter <= 3:
        debug_image = image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)
        runtime.draw_charuco_corners(debug_image, charuco_corners, charuco_ids)
        cv2.putText(debug_image, f"{len(object_points)} Punkte", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        debug_file = CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{debug_counter}.jpg"
        cv2.imwrite(str(debug_file), debug_image)
    # =======================
    
    return object_points, image_points, True


def calibrate():
    """Führt Kalibrierung durch."""
    # Setup
    board = create_charuco_board()
    dictionary = create_charuco_dictionary()
    runtime = create_charuco_runtime(board, dictionary)
    image_paths = load_all_images()
    
    # ===== ZUM ENTFERNEN ===== Diese Variable kann gelöscht werden, wenn Debug-Bilder nicht mehr nötig sind
    # In diesem Fall auch entfernen: if save_debug: ensure_output_directories() (Zeile 98)
    # und if save_debug: print(...) (Zeilen 152-153)
    save_debug = True  # ===== HIER ÄNDERN: True = Debug-Bilder speichern, False = nicht speichern
    # =======================
    
    print(f"Verwende: {runtime.description}")
    print(f"Bilder gefunden: {len(image_paths)}\n")
    
    if save_debug:
        ensure_output_directories()
    # =======================
    
    # Alle brauchbaren Punkte sammeln
    all_object_points = []
    all_image_points = []
    image_size = None
    usable_count = 0
    debug_counter = 0  # ===== Zähler für Debug-Bilder (nur erste 3)
    
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"✗ {image_path.name}: Bild konnte nicht geladen werden")
            continue
        
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        
        obj_pts, img_pts, success = detect_corners_in_image(
            image, board, runtime, image_path, save_debug, debug_counter
        )
        
        if success:
            all_object_points.append(obj_pts)
            all_image_points.append(img_pts)
            usable_count += 1
            debug_counter += 1  # ===== Counter nur für erfolgreiche Bilder erhöhen
            print(f"✓ {image_path.name}: {len(obj_pts)} Punkte erkannt")
        else:
            print(f"✗ {image_path.name}: Zu wenig Punkte oder keine Marker erkannt")
    
    if usable_count < 10:
        raise RuntimeError(f"Nur {usable_count} brauchbare Bilder - brauche mindestens 10!")
    
    print(f"\n{usable_count}/{len(image_paths)} Bilder verwendbar\n")
    
    # Kalibrierung
    if runtime.calibration_mode == "charuco":
        all_charuco_corners = []
        all_charuco_ids = []
        for obj_pts, img_pts in zip(all_object_points, all_image_points):
            all_charuco_corners.append(obj_pts)
            all_charuco_ids.append(img_pts)
        
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, image_size, None, None
        )
    else:
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_object_points, all_image_points, image_size, None, None
        )
    
    # RMS pro Bild ausgeben
    print("RMS-Fehler pro Bild:")
    for i, (obj_pts, img_pts) in enumerate(zip(all_object_points, all_image_points)):
        projected, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(img_pts, projected, cv2.NORM_L2)
        rms = np.sqrt((error ** 2) / len(projected))
        print(f"  Bild {i+1}: {rms:.6f}")
    
    # Gesamt-RMS
    print(f"\nGesamt-RMS: {retval:.6f}")
    
    # Ergebnisse speichern
    ensure_output_directories()
    np.save(str(CAMERA_MATRIX_PATH), camera_matrix)
    np.save(str(DIST_COEFFS_PATH), dist_coeffs)
    print(f"\nErgebnisse gespeichert in: {CALIBRATION_DEBUG_OUTPUT_DIR}")
    
    # ===== ZUM ENTFERNEN ===== Diese if-Bedingung kann gelöscht werden, wenn save_debug nicht mehr nötig ist
    if save_debug:
        print(f"Debug-Bilder unter: {CALIBRATION_DEBUG_OUTPUT_DIR}")
    # =======================


if __name__ == "__main__":
    try:
        calibrate()
    finally:
        cv2.destroyAllWindows()
