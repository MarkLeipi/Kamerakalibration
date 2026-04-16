from set_params import(
    create_charuco_dictionary,
    create_charuco_board,
    CAMERA_MATRIX_PATH,
    DIST_COEFFS_PATH,
    DEFAULT_WARP_ARUCO_PATH,
    UNDISTORTED_WARPING_IMAGE_PATH,
    HOMOGRAPHY_ARUCO_PATH,
    RECTIFIED_TOP_VIEW_PATH,
    RECTIFIED_TOP_VIEW_ORIGINAL
)

from opencv_support import(create_charuco_runtime)
import cv2
import numpy as np
from ArucoGen import(MARKER_LAYOUT)

MARKER_LAYOUT_MM = {
    0:  (0,  0),
    5:  (174, 0),

    12: (0,  61),
    23: (174, 61),

    42: (0,  122),
    87: (174, 122),
} #hangemessen


# 1. Setup & Daten laden (Marker-basiert)

dictionary = create_charuco_dictionary()
board = create_charuco_board()
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
runtime = create_charuco_runtime(board, dictionary)


matrix = np.load(CAMERA_MATRIX_PATH)
dist_coeffs = np.load(DIST_COEFFS_PATH)
image = cv2.imread(str(DEFAULT_WARP_ARUCO_PATH))

undistorted = cv2.undistort(image, matrix, dist_coeffs)

cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)



# 3. ArUco Marker detektieren (Marker-only)

marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(undistorted)

if marker_ids is None:
    print("Keine Marker gefunden")
else:
    print(f"Gefundene Marker: {marker_ids.flatten().tolist()}")

    # Marker einzeichnen (Debug!)
    debug_img = undistorted.copy()
    cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)

    cv2.imshow("Detected ArUco Markers", cv2.resize(debug_img, (800, 600)))
    cv2.waitKey(0)
    
    debug_img = undistorted.copy()
    cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
    cv2.imwrite(str(UNDISTORTED_WARPING_IMAGE_PATH), debug_img)


MARKER_SIZE_PX =  300 #handgemessen

#4. Berechnung der Homographie-Matrix
if marker_ids is None or len(marker_ids) < 2:
    raise RuntimeError("Zu wenige Marker für Homographie")
    

src_pts = []
dst_pts = []

for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
    if marker_id not in MARKER_LAYOUT_MM:
        continue

    # Bildpunkte (4 Ecken)
    img_corners = corners.reshape(4, 2).astype(np.float32)

    # Zielposition dieses Markers
    x, y = MARKER_LAYOUT_MM[marker_id]

    dst_corners = np.array([
        [x, y],
        [x + MARKER_SIZE_PX, y],
        [x + MARKER_SIZE_PX, y + MARKER_SIZE_PX],
        [x, y + MARKER_SIZE_PX],
    ], dtype=np.float32)

    src_pts.append(img_corners)
    dst_pts.append(dst_corners)


if len(src_pts) < 2:
    raise RuntimeError("Nicht genügend gültige Marker für Homographie")

src_pts = np.vstack(src_pts)
dst_pts = np.vstack(dst_pts)

# Homographie berechnen (RANSAC empfohlen)
H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)


# 5. Homographie Transformation anwenden
output_width = 1500
output_height = 1000
rectified = cv2.warpPerspective(
    undistorted,
    H,
    (output_width, output_height)
)

cv2.imshow("Top-View (Marker-based)", rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(str(RECTIFIED_TOP_VIEW_PATH), rectified)
cv2.imwrite(str(RECTIFIED_TOP_VIEW_ORIGINAL), image)
