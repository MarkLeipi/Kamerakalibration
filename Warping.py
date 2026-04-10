import cv2
import numpy as np
from set_params import (
    ARUCO_DICT,
    CAMERA_MATRIX_PATH,
    DEFAULT_WARP_IMAGE_PATH,
    DIST_COEFFS_PATH,
    HOMOGRAPHY_PATH,
    MARKER_LENGTH,
    PREVIEW_SIZE,
    RECTIFIED_TOP_VIEW_PATH,
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LENGTH,
    check_opencv_version,
)

# Prüfe beim Start, ob eine ausreichend neue OpenCV-Version installiert ist
check_opencv_version()


# Hilfsfunktion: Bild auf 16:9-Vorschaugröße verkleinern
def to_16_9_preview(image):
    return cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)


# Zeigt drei Vorschaufenster an: entzerrtes Bild, erkannte Marker und das Ergebnis-Draufsichtbild
def show_debug_preview_windows(undistorted_image, debug_image, rectified_image):
    cv2.imshow("Undistorted", to_16_9_preview(undistorted_image))
    cv2.imshow("Detections (debug)", to_16_9_preview(debug_image))
    cv2.imshow("Rectified top-view", to_16_9_preview(rectified_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Hauptfunktion: Berechnet die Homographie-Matrix (Draufsicht-Transformation)
# aus einem Messbild mit ChArUco-Board und speichert das entzerrte Draufsichtbild.
def compute_topview_homography(
    image_path=str(DEFAULT_WARP_IMAGE_PATH),
    camera_matrix_path=None,
    dist_coeffs_path=None,
    output_image_path=str(RECTIFIED_TOP_VIEW_PATH),
    homography_path=str(HOMOGRAPHY_PATH),
    pixels_per_meter=2000.0,
    show_debug_previews=True,
):
    if camera_matrix_path is None:
        camera_matrix_path = str(CAMERA_MATRIX_PATH)
    if dist_coeffs_path is None:
        dist_coeffs_path = str(DIST_COEFFS_PATH)

    # Gespeicherte Kalibrierungsdaten laden
    camera_matrix = np.load(camera_matrix_path)
    dist_coeffs = np.load(dist_coeffs_path)

    # Messbild laden und auf Verzerrung prüfen
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: '{image_path}'")

    # Linsenverzerrung aus dem Bild entfernen
    un_img = cv2.undistort(img, camera_matrix, dist_coeffs)

    # ChArUco-Board definieren (muss identisch mit dem gedruckten Board sein)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    all_board_corners = board.getChessboardCorners()

    # ArUco-Marker im entzerrten Bild erkennen
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, _ = detector.detectMarkers(un_img)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("Keine ArUco-Marker im Bild gefunden!")

    # Erkannte Marker zur Kontrolle einzeichnen
    debug_draw = un_img.copy()
    cv2.aruco.drawDetectedMarkers(debug_draw, marker_corners, marker_ids)

    # Aus den erkannten Markern die Schachbrett-Ecken (ChArUco-Ecken) berechnen
    # (Die ersten beiden Rückgabewerte sind immer Ecken und IDs)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    detected = charuco_detector.detectBoard(un_img)
    charuco_corners, charuco_ids = detected[0], detected[1]
    retval = 0 if charuco_ids is None else len(charuco_ids)

    if charuco_ids is None or retval < 4:
        raise RuntimeError("Nicht genug ChArUco-Ecken erkannt (mindestens 4 nötig)!")

    # Bildpunkte (2D) den bekannten Board-Punkten (2D in Metern) zuordnen
    img_pts = []
    board_pts = []
    for i in range(len(charuco_ids)):
        cid = int(charuco_ids[i][0])
        if cid < 0 or cid >= len(all_board_corners):
            continue
        board_xy = all_board_corners[cid][:2]
        img_xy = charuco_corners[i][0]
        img_pts.append(img_xy)
        board_pts.append(board_xy)

    if len(img_pts) < 4:
        raise RuntimeError("Weniger als 4 gültige ChArUco-Ecken nach dem Filtern.")

    img_pts = np.array(img_pts, dtype=np.float32)
    board_pts = np.array(board_pts, dtype=np.float32)
    # Board-Koordinaten von Metern in Pixel umrechnen
    board_pts_px = board_pts * pixels_per_meter

    # Homographie berechnen: Transformation vom Kamerabild in die Draufsicht
    H, inliers = cv2.findHomography(
        img_pts,
        board_pts_px,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )
    if H is None:
        raise RuntimeError("Homographie-Berechnung fehlgeschlagen.")

    # Homographie mit nur den als gut eingestuften Punkten verfeinern
    if inliers is not None and inliers.sum() >= 4 and inliers.sum() < len(inliers):
        img_pts_in = img_pts[inliers.ravel() == 1]
        board_pts_px_in = board_pts_px[inliers.ravel() == 1]
        H_refined, _ = cv2.findHomography(img_pts_in, board_pts_px_in, method=0)
        if H_refined is not None:
            H = H_refined

    # Ausgabegröße des Draufsichtbilds berechnen
    squaresX, squaresY = board.getChessboardSize()
    square_length_m = board.getSquareLength()
    w = int(round(squaresX * square_length_m * pixels_per_meter))
    h = int(round(squaresY * square_length_m * pixels_per_meter))

    # Bild in die Draufsicht transformieren
    rectified = cv2.warpPerspective(un_img, H, (w, h))

    cv2.putText(
        debug_draw,
        f"ChArUco corners: {len(img_pts)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if show_debug_previews:
        show_debug_preview_windows(un_img, debug_draw, rectified)

    # Ergebnisse speichern
    cv2.imwrite(output_image_path, rectified)
    np.save(homography_path, H)

    return {
        "homography": H,
        "homography_path": homography_path,
        "rectified_path": output_image_path,
        "rectified_size": (w, h),
        "charuco_corner_count": int(len(img_pts)),
    }


# Wird nur ausgeführt, wenn diese Datei direkt gestartet wird (nicht bei Import)
if __name__ == "__main__":
    result = compute_topview_homography()
    print(f"Ausgabeauflösung = {result['rectified_size'][0]} x {result['rectified_size'][1]} px")
