"""Entzerrung und Top-View-Warping mit klarer Schritt-fuer-Schritt-Struktur."""

from pathlib import Path

import cv2
import numpy as np

from opencv_support import create_charuco_runtime
from set_params import (
    CAMERA_MATRIX_PATH,
    DEFAULT_WARP_IMAGE_PATH,
    DIST_COEFFS_PATH,
    HOMOGRAPHY_PATH,
    PREVIEW_SIZE,
    RECTIFIED_TOP_VIEW_PATH,
    create_charuco_board,
    create_charuco_dictionary,
    ensure_output_directories,
)


def resize_for_preview(image):
    """Skaliert ein Bild fuer die Debug-Fenster auf eine feste Groesse."""
    return cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)


def show_debug_preview_windows(undistorted_image, debug_image, rectified_image):
    """Zeigt Zwischenstand und Endergebnis nebeneinander an."""
    cv2.imshow("Undistorted", resize_for_preview(undistorted_image))
    cv2.imshow("Detections (debug)", resize_for_preview(debug_image))
    cv2.imshow("Rectified top-view", resize_for_preview(rectified_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_numpy_array(path, description):
    """Laedt eine gespeicherte NumPy-Datei mit klarer Fehlermeldung."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} wurde nicht gefunden: '{file_path}'")
    return np.load(str(file_path))


def read_image_or_raise(image_path):
    """Laedt das Bild fuer das Warping oder stoppt mit einer klaren Meldung."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Bild wurde nicht gefunden: '{image_path}'")
    return image


def detect_board_points(undistorted_image, board, runtime):
    """
    Sucht das Board im bereits entzerrten Bild und liefert passende Punktpaare.

    Wir brauchen zwei Mengen von Punkten:
    - Bildpunkte: Wo liegen die erkannten ChArUco-Ecken im Foto?
    - Board-Punkte: Wo liegen dieselben Ecken im ebenen Koordinatensystem
      des gedruckten Boards?
    """
    marker_corners, marker_ids, _ = runtime.detect_markers(undistorted_image)
    marker_count = 0 if marker_ids is None else len(marker_ids)

    if marker_count == 0:
        raise RuntimeError("Im Warping-Bild wurden keine ArUco-Marker erkannt.")

    debug_image = undistorted_image.copy()
    cv2.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)

    charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(
        undistorted_image,
        marker_corners,
        marker_ids,
    )

    if charuco_count < 4 or charuco_corners is None or charuco_ids is None:
        raise RuntimeError(
            "Im Warping-Bild wurden nicht genug ChArUco-Ecken erkannt "
            "(mindestens 4 werden benoetigt)."
        )

    runtime.draw_charuco_corners(debug_image, charuco_corners, charuco_ids)

    object_points, image_points = board.matchImagePoints(
        charuco_corners,
        charuco_ids,
    )

    if object_points is None or image_points is None or len(object_points) < 4:
        raise RuntimeError(
            "Die erkannten ChArUco-Ecken konnten nicht in stabile Punktpaare "
            "fuer das Warping umgewandelt werden."
        )

    board_plane_points_m = object_points.reshape(-1, 3)[:, :2].astype(np.float32)
    image_points_px = image_points.reshape(-1, 2).astype(np.float32)

    return debug_image, image_points_px, board_plane_points_m


def compute_rectified_output_size(board, pixels_per_meter):
    """Berechnet die Groesse des Top-View-Bildes aus Boardgroesse und Aufloesung."""
    squares_x, squares_y = board.getChessboardSize()
    square_length_m = board.getSquareLength()

    width_px = int(round(squares_x * square_length_m * pixels_per_meter))
    height_px = int(round(squares_y * square_length_m * pixels_per_meter))
    return width_px, height_px


def compute_topview_homography(
    image_path=str(DEFAULT_WARP_IMAGE_PATH),
    camera_matrix_path=str(CAMERA_MATRIX_PATH),
    dist_coeffs_path=str(DIST_COEFFS_PATH),
    output_image_path=str(RECTIFIED_TOP_VIEW_PATH),
    homography_path=str(HOMOGRAPHY_PATH),
    pixels_per_meter=2000.0,
    show_debug_previews=True,
):
    """
    Berechnet eine Homographie fuer die Draufsicht auf das ChArUco-Board.

    Vereinfacht gesagt:
    - Die Kamerakalibrierung entfernt zuerst die Linsenverzerrung.
    - Danach wird aus erkannten Board-Ecken berechnet, wie das Bild
      aussehen muss, damit die Ebene des Boards frontal von oben wirkt.
    """
    board = create_charuco_board()
    dictionary = create_charuco_dictionary()
    runtime = create_charuco_runtime(board, dictionary)

    camera_matrix = load_numpy_array(camera_matrix_path, "Kameramatrix")
    dist_coeffs = load_numpy_array(dist_coeffs_path, "Verzerrungskoeffizienten")

    input_image = read_image_or_raise(image_path)
    undistorted_image = cv2.undistort(input_image, camera_matrix, dist_coeffs)

    print(f"Verwende {runtime.description}")

    debug_image, image_points_px, board_plane_points_m = detect_board_points(
        undistorted_image,
        board,
        runtime,
    )

    # Das Board liegt in Metern vor. Fuer das Zielbild rechnen wir die
    # metrischen Koordinaten in Pixel um, damit cv2.warpPerspective direkt
    # ein Bild mit passender Aufloesung erzeugen kann.
    board_plane_points_px = board_plane_points_m * pixels_per_meter

    homography_matrix, inliers = cv2.findHomography(
        image_points_px,
        board_plane_points_px,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )

    if homography_matrix is None:
        raise RuntimeError("Die Homographie konnte nicht berechnet werden.")

    # Nach dem robusten RANSAC-Schritt wird die Matrix mit den Inliern noch
    # einmal sauber nachgerechnet. Das behaelt das bisherige Verhalten bei,
    # ist aber jetzt klarer als eigener Schritt sichtbar.
    if inliers is not None and inliers.sum() >= 4 and inliers.sum() < len(inliers):
        inlier_mask = inliers.ravel() == 1
        refined_homography, _ = cv2.findHomography(
            image_points_px[inlier_mask],
            board_plane_points_px[inlier_mask],
            method=0,
        )
        if refined_homography is not None:
            homography_matrix = refined_homography

    output_width_px, output_height_px = compute_rectified_output_size(
        board,
        pixels_per_meter,
    )
    rectified_image = cv2.warpPerspective(
        undistorted_image,
        homography_matrix,
        (output_width_px, output_height_px),
    )

    cv2.putText(
        debug_image,
        f"Charuco corners: {len(image_points_px)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    ensure_output_directories()
    cv2.imwrite(str(output_image_path), rectified_image)
    np.save(str(homography_path), homography_matrix)

    if show_debug_previews:
        show_debug_preview_windows(undistorted_image, debug_image, rectified_image)

    return {
        "homography": homography_matrix,
        "homography_path": str(homography_path),
        "rectified_path": str(output_image_path),
        "rectified_size": (output_width_px, output_height_px),
        "charuco_corner_count": int(len(image_points_px)),
    }


if __name__ == "__main__":
    result = compute_topview_homography()
    print(
        "Output rectified resolution = "
        f"{result['rectified_size'][0]} x {result['rectified_size'][1]} px"
    )
