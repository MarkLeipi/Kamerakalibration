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


def load_all_images():
    """Laden aller Kalibrierbilder."""
    if not CAL_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {CAL_IMAGES_DIR}")
    
    supported = {".jpg", ".jpeg", ".png"}
    images = sorted(
        path for path in CAL_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )
    return images


def detect_corners_in_image(image, board, runtime, save_debug, debug_counter):
    """
    Erkennt ChArUco-Ecken in einem Bild.
    Rückgabe: (charuco_corners, charuco_ids, erfolgreich)
    """
    # Marker erkennen
    marker_corners, marker_ids, _ = runtime.detect_markers(image)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, False
    
    # ChArUco-Ecken aus Markern extrahieren
    charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(
        image, marker_corners, marker_ids
    )
    
    # Mindestens 4 Ecken nötig für eine gültige Schätzung
    if charuco_count < 4 or charuco_corners is None or charuco_ids is None:
        return None, None, False
    
    # Optional: Debug-Bild speichern
    if save_debug and debug_counter < NMB_DEBUG:
        debug_image = image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids)
        
        debug_file = CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{debug_counter}_detect.jpg"
        cv2.imwrite(str(debug_file), debug_image)
    
    return charuco_corners, charuco_ids, True


def calibrate():
    """Führt Kalibrierung durch."""
    ensure_output_directories()
    
    # Setup
    board = create_charuco_board()
    dictionary = create_charuco_dictionary()
    runtime = create_charuco_runtime(board, dictionary)
    image_paths = load_all_images()
    
    all_charuco_corners = []
    all_charuco_ids = []
    usable_images = []
    image_size = None
    
    print(f"Verwende: {runtime.description}")
    print(f"Bilder gefunden: {len(image_paths)}\n")
    
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        
        c_corners, c_ids, success = detect_corners_in_image(
            image, board, runtime, SAVE_DEBUG, len(usable_images)
        )
        
        if success:
            all_charuco_corners.append(c_corners)
            all_charuco_ids.append(c_ids)
            usable_images.append(image)
            print(f"✓ {image_path.name}: {len(c_corners)} Ecken erkannt")
        else:
            print(f"✗ {image_path.name}: Zu wenig Punkte erkannt")
    
    if len(all_charuco_corners) < 10:
        raise RuntimeError(f"Nur {len(all_charuco_corners)} brauchbare Bilder - brauche mindestens 10!")
    
    print(f"\n{len(all_charuco_corners)}/{len(image_paths)} Bilder verwendbar\n")
    
    # Kalibrierung durchführen
    # In neueren OpenCV-Versionen ist der Aufruf über den Namespace cv2.aruco korrekt, 
    # wenn opencv-contrib-python installiert ist.
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    
    print(f"Gesamt-RMS: {retval:.6f}\n")
    print("RMS-Fehler pro Bild:")

    for i, (c_corners, c_ids) in enumerate(zip(all_charuco_corners, all_charuco_ids)):
        # Für den RMS pro Bild brauchen wir die Entsprechung von 3D zu 2D Punkten
        obj_pts, img_pts = board.matchImagePoints(c_corners, c_ids)
        
        # Reprojektion
        projected, _ = cv2.projectPoints(
            obj_pts, rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )

        error = cv2.norm(img_pts, projected, cv2.NORM_L2)
        rms = np.sqrt((error ** 2) / len(projected))
        print(f"  Bild {i+1}: {rms:.6f}")

        # Optional: Undistorted Debug-Bild speichern
        if SAVE_DEBUG and i < NMB_DEBUG:
            undistorted = cv2.undistort(usable_images[i], camera_matrix, dist_coeffs)
            out_file = CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i+1}_undistorted.jpg"
            cv2.imwrite(str(out_file), undistorted)
    
    # Ergebnisse speichern
    np.save(str(CAMERA_MATRIX_PATH), camera_matrix)
    np.save(str(DIST_COEFFS_PATH), dist_coeffs)
    
    print(f"\nErgebnisse gespeichert unter: {CAMERA_MATRIX_PATH.parent}")
    if SAVE_DEBUG:
        print(f"Debug-Bilder unter: {CALIBRATION_DEBUG_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        calibrate()
    finally:
        cv2.destroyAllWindows()
