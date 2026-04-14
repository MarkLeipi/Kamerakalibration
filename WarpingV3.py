import cv2
import numpy as np
from opencv_support import create_charuco_runtime
from set_params import (
    CAMERA_MATRIX_PATH, DEFAULT_WARP_IMAGE_PATH, DIST_COEFFS_PATH,
    create_charuco_board, create_charuco_dictionary
)

# 1. Setup & Daten laden
board = create_charuco_board()
dictionary = create_charuco_dictionary()
runtime = create_charuco_runtime(board, dictionary)

matrix = np.load(CAMERA_MATRIX_PATH)
dist_coeffs = np.load(DIST_COEFFS_PATH)
image = cv2.imread(str(DEFAULT_WARP_IMAGE_PATH))

# 2. Entzerren (Undistort)
undistorted = cv2.undistort(image, matrix, dist_coeffs)

# 3. ChArUco Detektion
marker_corners, marker_ids, _ = runtime.detect_markers(undistorted)
charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(
    undistorted, marker_corners, marker_ids
)

if charuco_count >= 4:
    # 4. Punkte matchen & Homographie berechnen
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    
    # Reduktion auf 2D (Board-Ebene) und Skalierung für das Zielbild (z.B. 2000 px/m)
    src_pts = img_pts.reshape(-1, 2).astype(np.float32)
    dst_pts = obj_pts.reshape(-1, 3)[:, :2].astype(np.float32) * 2000.0
    
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # 5. Warping (Draufsicht erzeugen)
    # Größe basierend auf Board-Dimensionen berechnen
    w_m, h_m = board.getChessboardSize()[0] * board.getSquareLength(), board.getChessboardSize()[1] * board.getSquareLength()
    rectified = cv2.warpPerspective(undistorted, H, (int(w_m * 2000), int(h_m * 2000)))

    # 6. Anzeige
    cv2.imshow("Top-View", cv2.resize(rectified, (800, 600)))
    cv2.waitKey(0)
else:
    print(f"Fehler: Nur {charuco_count} Ecken gefunden. Brauche mind. 4.")

cv2.destroyAllWindows()
