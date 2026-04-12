"""Kamerakalibrierung mit ChArUco-Board."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from opencv_support import create_charuco_runtime
from set_params import (
    CAL_IMAGES_DIR,
    CALIBRATION_DEBUG_OUTPUT_DIR,
    CAMERA_MATRIX_PATH,
    DIST_COEFFS_PATH,
    PREVIEW_SIZE,
    DEBUG_TIME,
    create_charuco_board,
    create_charuco_dictionary,
    ensure_output_directories,
)


@dataclass
class CalibrationObservation:
    """Speichert alle Messdaten, die aus genau einem Bild stammen."""

    image_path: Path
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray


def list_calibration_images():
    """Sammelt alle Bilddateien aus dem Kalibrierordner in stabiler Reihenfolge."""
    if not CAL_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Der Kalibrierordner wurde nicht gefunden: '{CAL_IMAGES_DIR}'"
        )

    supported_suffixes = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        path
        for path in CAL_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    return image_paths


def read_image_or_raise(image_path: Path):
    """Laedt ein Bild oder stoppt mit einer klaren Meldung."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Bild konnte nicht geladen werden: '{image_path}'")
    return image



def save_detection_debug_image(
    debug_image,
    image_path,
    status,
    marker_count,
    charuco_count,
    matched_point_count,
):
    """
    Speichert ein Debug-Bild fuer die spaetere Fehlersuche.

    So kann man nach einem Lauf direkt sehen, welches Bild brauchbar war
    und an welchem Schritt ein Bild ausgeschieden ist.
    """
    ensure_output_directories()

    file_name = (
        f"{status}__markers_{marker_count:02d}"
        f"__charuco_{charuco_count:02d}"
        f"__pairs_{matched_point_count:02d}"
        f"__{image_path.stem}.jpg"
    )
    output_path = CALIBRATION_DEBUG_OUTPUT_DIR / file_name
    cv2.imwrite(str(output_path), debug_image)


def collect_calibration_observations(
    board,
    runtime,
    image_paths,
    save_debug_images,
):
    """
    Liest alle Kalibrierbilder ein und sammelt nur die Bilder, die genug
    verwertbare ChArUco-Ecken fuer die Kalibrierung liefern.
    """
    observations = []
    image_size = None
    status_counts = {
        "usable": 0,
        "no_markers": 0,
        "too_few_charuco_corners": 0,
        "match_points_failed": 0,
    }

    for image_path in image_paths:
        image = read_image_or_raise(image_path)
        debug_image = image.copy()
        matched_point_count = 0
        status = "no_markers"

        if image_size is None:
            # OpenCV erwartet die Bildgroesse in der Reihenfolge (Breite, Hoehe).
            image_size = (image.shape[1], image.shape[0])

        marker_corners, marker_ids, _ = runtime.detect_markers(image)
        marker_count = 0 if marker_ids is None else len(marker_ids)

        if marker_count > 0:
            cv2.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)
            status = "too_few_charuco_corners"

        charuco_count, charuco_corners, charuco_ids = runtime.detect_charuco_corners(
            image,
            marker_corners,
            marker_ids,
        )

        if charuco_count > 0:
            runtime.draw_charuco_corners(debug_image, charuco_corners, charuco_ids)

        if charuco_count >= 4 and charuco_corners is not None and charuco_ids is not None:
            # matchImagePoints ordnet jeder 2D-Bildmessung den passenden 3D-Punkt
            # auf dem Board zu. Genau diese Paare braucht die Kalibrierung spaeter.
            status = "match_points_failed"
            object_points, image_points = board.matchImagePoints(
                charuco_corners,
                charuco_ids,
            )

            if object_points is not None:
                matched_point_count = len(object_points)

            if (
                object_points is not None
                and image_points is not None
                and len(object_points) >= 4
            ):
                status = "usable"
                observations.append(
                    CalibrationObservation(
                        image_path=image_path,
                        charuco_corners=charuco_corners,
                        charuco_ids=charuco_ids,
                        object_points=object_points,
                        image_points=image_points,
                    )
                )

        status_counts[status] += 1

        cv2.putText(
            debug_image,
            f"Status: {status} | Pairs: {matched_point_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if status == "usable" else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if save_debug_images:
            save_detection_debug_image(
                debug_image,
                image_path,
                status,
                marker_count,
                charuco_count,
                matched_point_count,
            )

    if not observations:
        if save_debug_images:
            print(
                "\nDebug-Bilder wurden gespeichert unter: "
                f"{CALIBRATION_DEBUG_OUTPUT_DIR}"
            )
        raise RuntimeError(
            "In keinem Kalibrierbild wurden mindestens 4 nutzbare "
            "ChArUco-Ecken gefunden."
        )

    return observations, image_size


def calibrate_camera(board, runtime, observations, image_size):
    """Fuehrt die eigentliche Kamerakalibrierung mit der gewaehlten API aus."""
    all_charuco_corners = [item.charuco_corners for item in observations]
    all_charuco_ids = [item.charuco_ids for item in observations]
    object_points = [item.object_points for item in observations]
    image_points = [item.image_points for item in observations]

    if runtime.calibration_mode == "charuco":
        return cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None,
        )

    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )


def print_per_image_rms(observations, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Gibt den Reprojektionsfehler pro Bild aus.

    Vereinfacht gesagt: OpenCV berechnet aus den gefundenen Kalibrierdaten,
    wo die Board-Punkte im Bild liegen sollten. Danach wird verglichen, wie
    weit diese berechneten Punkte von den gemessenen Bildpunkten abweichen.
    Je kleiner der RMS-Wert, desto besser passt das Modell zu diesem Bild.
    """
    print("\nRMS reprojection error pro verwendetem Bild:")

    for index, observation in enumerate(observations):
        projected_points, _ = cv2.projectPoints(
            observation.object_points,
            rvecs[index],
            tvecs[index],
            camera_matrix,
            dist_coeffs,
        )

        error = cv2.norm(observation.image_points, projected_points, cv2.NORM_L2)
        rms = np.sqrt((error ** 2) / len(projected_points))
        print(f"{observation.image_path.name}: {rms:.6f}")


def save_calibration_results(camera_matrix, dist_coeffs):
    """Speichert die Kalibrierergebnisse in die vorgesehenen Projektordner."""
    ensure_output_directories()
    np.save(str(CAMERA_MATRIX_PATH), camera_matrix)
    np.save(str(DIST_COEFFS_PATH), dist_coeffs)


def calibrate_and_save_parameters(
    save_debug_images=False,
):
    """
    Fuehrt die komplette Kamerakalibrierung durch und speichert das Ergebnis.

    Ablauf in einfachen Worten:
    1. Alle Kalibrierbilder werden geladen.
    2. In jedem Bild werden ArUco-Marker und daraus ChArUco-Ecken gesucht.
    3. Nur Bilder mit genug brauchbaren Ecken werden fuer die Kalibrierung genutzt.
    4. OpenCV berechnet Kameramatrix und Verzerrungskoeffizienten.
    5. Die Ergebnisse werden gespeichert und auf Wunsch in Vorschaufenstern gezeigt.

    Fuer die Fehlersuche koennen zusaetzlich pro Bild Diagnosewerte in die
    Konsole geschrieben und Debug-Bilder gespeichert werden.
    """
    board = create_charuco_board()
    dictionary = create_charuco_dictionary()
    runtime = create_charuco_runtime(board, dictionary)
    image_paths = list_calibration_images()

    print(f"Verwende {runtime.description}")

    try:
        observations, image_size = collect_calibration_observations(
            board,
            runtime,
            image_paths,
            save_debug_images,
        )

        retval, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            board,
            runtime,
            observations,
            image_size,
        )

        print_per_image_rms(
            observations,
            rvecs,
            tvecs,
            camera_matrix,
            dist_coeffs,
        )
        print(f"\nOverall RMS reprojection error: {retval:.6f}")

        save_calibration_results(camera_matrix, dist_coeffs)

        return {
            "camera_matrix_path": str(CAMERA_MATRIX_PATH),
            "dist_coeffs_path": str(DIST_COEFFS_PATH),
            "debug_output_dir": str(CALIBRATION_DEBUG_OUTPUT_DIR),
            "image_count": len(image_paths),
            "detected_frame_count": len(observations),
            "rms": float(retval),
        }
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    result = calibrate_and_save_parameters()
    print("RMS:", result["rms"])
    print("used calibration pictures:", result["image_count"])
    print("Detected frames:", result["detected_frame_count"])
    print("camera_matrix:", np.load(CAMERA_MATRIX_PATH))
    print("distortion coefficient:", np.load(DIST_COEFFS_PATH))
