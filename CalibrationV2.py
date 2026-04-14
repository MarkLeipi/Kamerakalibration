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

    all_obj_pts, all_img_pts, used_images = [], [], []
    image_size = None

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None: continue
        if image_size is None: image_size = (img.shape[1], img.shape[0])

        corners, ids, _ = runtime.detect_markers(img)
        count, c_corners, c_ids = runtime.detect_charuco_corners(img, corners, ids)

        if count >= 4:
            obj_pts, img_pts = board.matchImagePoints(c_corners, c_ids)
            all_obj_pts.append(obj_pts)
            all_img_pts.append(img_pts)
            used_images.append(img)
            print(f"✓ {path.name}: {len(obj_pts)} Punkte erkannt")
            
            # Debug-Bilder speichern (falls aktiv)
            if SAVE_DEBUG and i < NMB_DEBUG:
                debug_img = img.copy()
                runtime.draw_charuco_corners(debug_img, c_corners, c_ids)
                cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i}.jpg"), debug_img)
        else:
            print(f"✗ {path.name}: Zu wenig Punkte")

    if len(all_obj_pts) < 10:
        raise RuntimeError(f"Nur {len(all_obj_pts)} Bilder brauchbar - brauche 10!")

    # Kalibrierung (Gesamt-RMS ist 'retval')
    retval, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_obj_pts, all_img_pts, board, image_size, None, None
    )

    print(f"\n{len(all_obj_pts)}/{len(image_paths)} Bilder verwendbar\n\nRMS-Fehler pro Bild:")
    
    # RMS pro Bild & Undistorted Debug
    for i, (obj, img_pt) in enumerate(zip(all_obj_pts, all_img_pts)):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, D)
        rms = np.sqrt(np.mean(np.sum((img_pt.reshape(-1, 2) - proj.reshape(-1, 2))**2, axis=1)))
        print(f"  Bild {i+1}: {rms:.6f}")

        if SAVE_DEBUG and i < NMB_DEBUG:
            undist = cv2.undistort(used_images[i], K, D)
            cv2.imwrite(str(CALIBRATION_DEBUG_OUTPUT_DIR / f"Bild{i+1}_undistorted.jpg"), undist)

    # Speichern
    np.save(str(CAMERA_MATRIX_PATH), K)
    np.save(str(DIST_COEFFS_PATH), D)
    print(f"\nErgebnisse gespeichert in: {CALIBRATION_DEBUG_OUTPUT_DIR}")
    if SAVE_DEBUG: print(f"Debug-Bilder unter: {CALIBRATION_DEBUG_OUTPUT_DIR}")

if __name__ == "__main__":
    calibrate()
