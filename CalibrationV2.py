import cv2
import numpy as np
from set_params import (
    CAL_IMAGES_DIR, CALIBRATION_DEBUG_OUTPUT_DIR, CAMERA_MATRIX_PATH,
    DIST_COEFFS_PATH, SAVE_DEBUG, NMB_DEBUG, 
    create_charuco_board, create_charuco_dictionary, ensure_output_directories
)
from opencv_support import create_charuco_runtime

def calibrate():
    board = create_charuco_board()
    runtime = create_charuco_runtime(board, create_charuco_dictionary())
    ensure_output_directories()
    
    image_paths = sorted([p for p in CAL_IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Verwende: {runtime.description}\nBilder gefunden: {len(image_paths)}\n")

    all_charuco_corners, all_charuco_ids, used_images = [], [], []
    image_size = None

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None: continue
        if image_size is None: image_size = (img.shape[1], img.shape[0])

        # Marker und dann ChArUco-Ecken erkennen
        corners, ids, _ = runtime.detect_markers(img)
        count, c_corners, c_ids = runtime.detect_charuco_corners(img, corners, ids)

        if count >= 4:
            all_charuco_corners.append(c_corners)
            all_charuco_ids.append(c_ids)
            used_images.append(img)
            print(f"✓ {path.name}: {count} ChArUco-Ecken erkannt")
            
            if SAVE_DEBUG and i < NMB_DEBUG:
                debug_img = img.copy()
                cv2.aruco.drawDetectedCornersCharuco(debug_img, c_corners, c_ids)
                cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i}.jpg"), debug_img)
        else:
            print(f"✗ {path.name}: Zu wenig Punkte")

    if len(all_charuco_corners) < 10:
        raise RuntimeError(f"Nur {len(all_charuco_corners)} Bilder brauchbar - brauche 10!")

    # DER FIX: calibrateCameraCharuco Aufruf je nach OpenCV Version
    # Wir nutzen die Methode des CharucoBoard Moduls oder die globale Funktion
    res = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    retval, K, D, rvecs, tvecs = res

    print(f"\n{len(all_charuco_corners)}/{len(image_paths)} Bilder verwendbar\nGesamt-RMS: {retval:.6f}\n\nRMS-Fehler pro Bild:")
    
    for i, (c_corn, c_id) in enumerate(zip(all_charuco_corners, all_charuco_ids)):
        # Reprojektion zur Fehlerberechnung
        obj_pts, img_pts = board.matchImagePoints(c_corn, c_id)
        proj, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], K, D)
        rms = np.sqrt(np.mean(np.sum((img_pts.reshape(-1, 2) - proj.reshape(-1, 2))**2, axis=1)))
        print(f"  Bild {i+1}: {rms:.6f}")

        if SAVE_DEBUG and i < NMB_DEBUG:
            undist = cv2.undistort(used_images[i], K, D)
            cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i+1}_undistorted.jpg"), undist)

    np.save(str(CAMERA_MATRIX_PATH), K)
    np.save(str(DIST_COEFFS_PATH), D)
    print(f"\nErgebnisse gespeichert in: {CALIBRATION_DEBUG_OUTPUT_DIR}")

if __name__ == "__main__":
    calibrate()
