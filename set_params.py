# Definition der globalen Parameter und Pfade, die in allen Modulen verwendet werden.


from pathlib import Path
import cv2


# Definition der Pfade
BASE_DIR = Path(__file__).resolve().parent


# Eingabe von Kalibrier und Warping Bildern
CAL_IMAGES_DIR = BASE_DIR / "Cal_5x7"
WARP_IMAGES_DIR = BASE_DIR / "Warp_Images"


# Ausgabedaten
OUTPUTS_DIR = BASE_DIR / "outputs"
CALIBRATION_OUTPUT_DIR = OUTPUTS_DIR / "calibration"
CALIBRATION_DEBUG_OUTPUT_DIR = OUTPUTS_DIR / "calibration_debug"
HOMOGRAPHY_OUTPUT_DIR = OUTPUTS_DIR / "homography"
RECTIFIED_OUTPUT_DIR = OUTPUTS_DIR / "rectified"


# Einzelne Dateien
CHARUCO_BOARD_IMAGE_PATH = BASE_DIR / "ChArUco_Marker.png"
DEFAULT_WARP_IMAGE_PATH = WARP_IMAGES_DIR / "1.jpg"
CAMERA_MATRIX_PATH = CALIBRATION_OUTPUT_DIR / "camera_matrix.npy"
DIST_COEFFS_PATH = CALIBRATION_OUTPUT_DIR / "dist_coeffs.npy"
HOMOGRAPHY_PATH = HOMOGRAPHY_OUTPUT_DIR / "H_charuco_topview.npy"
RECTIFIED_TOP_VIEW_PATH = RECTIFIED_OUTPUT_DIR / "Rectified_top_view.jpg"


# ChArUco-Checkerboard Definition 
ARUCO_DICT_NAME = "DICT_6X6_250"

# OpenCV erwartet die Brettgroesse in der Reihenfolge (x, y):
# x  Spalten
# y  Zeilen
BOARD_SQUARE_COUNT_X = 5
BOARD_SQUARE_COUNT_Y = 7

# Reale Groesse eines Feldes bzw. eines ArUco-Markers in Metern.
BOARD_SQUARE_LENGTH_M = 0.02
BOARD_MARKER_LENGTH_M = 0.01


# Groesse des generierten Board-Bildes fuer den Ausdruck.
BOARD_IMAGE_WIDTH_PX = 640
BOARD_IMAGE_MARGIN_PX = 20


# Einheitliche Vorschaubilder fuer Debug-Fenster.
PREVIEW_SIZE = (600, 800)
DEBUG_TIME = 20


def create_charuco_dictionary():
    """Erzeugt das im ganzen Projekt verwendete ArUco-Woerterbuch."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "In dieser OpenCV-Installation fehlt das aruco-Modul. "
            "Bitte installiere opencv-contrib-python 4.x."
        )

    if not hasattr(cv2.aruco, ARUCO_DICT_NAME):
        raise RuntimeError(
            f"Das angeforderte ArUco-Woerterbuch '{ARUCO_DICT_NAME}' "
            "wird von dieser OpenCV-Version nicht unterstuetzt."
        )

    dictionary_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_charuco_board():
    """Erzeugt das ChArUco-Board mit genau den Projektparametern."""
    return cv2.aruco.CharucoBoard(
        (BOARD_SQUARE_COUNT_X, BOARD_SQUARE_COUNT_Y),
        BOARD_SQUARE_LENGTH_M,
        BOARD_MARKER_LENGTH_M,
        create_charuco_dictionary(),
    )


def ensure_output_directories():
    """Legt die benoetigten Ausgabeordner an, falls sie noch fehlen."""
    for directory in (
        OUTPUTS_DIR,
        CALIBRATION_OUTPUT_DIR,
        CALIBRATION_DEBUG_OUTPUT_DIR,
        HOMOGRAPHY_OUTPUT_DIR,
        RECTIFIED_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
