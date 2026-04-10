##########################################################
# [set_params.py]
# Feste Parameter/ Speicherorte etc. 
##########################################################


from pathlib import Path

import cv2

# -- FILE MANAGEMENT --
BASE_DIR = Path(__file__).resolve().parent 
CAL_IMAGES_DIR = BASE_DIR / 'Cal_Images2' # Ordnername der Kalibrationsbilder
WARP_IMAGES_DIR = BASE_DIR / 'Warp_Images' # Ordner mit den Messaufnahmen, die entzerrt werden müssen 
OUTPUTS_DIR = BASE_DIR / 'outputs' # genereller Output-Ordner
CALIBRATION_OUTPUT_DIR = OUTPUTS_DIR / 'calibration' # Kameramatrix
HOMOGRAPHY_OUTPUT_DIR = OUTPUTS_DIR / 'homography' # Homographie/ Warping-Trafo Matrix
RECTIFIED_OUTPUT_DIR = OUTPUTS_DIR / 'rectified' # entzerrtes 2D-Bild


CHARUCO_BOARD_IMAGE_PATH = BASE_DIR / 'ChArUco_Marker.png' # Speichername des Checkerboards
DEFAULT_WARP_IMAGE_PATH = WARP_IMAGES_DIR / 'IMG_0237.jpg' # Dateiname des zu warpenden Bildes 
CAMERA_MATRIX_PATH = CALIBRATION_OUTPUT_DIR / 'camera_matrix.npy' # Speicherort der Kameramatrix (3x3)
DIST_COEFFS_PATH = CALIBRATION_OUTPUT_DIR / 'dist_coeffs.npy' # Speicherort der Verzerrungskoeffizienten
HOMOGRAPHY_PATH = HOMOGRAPHY_OUTPUT_DIR / 'H_charuco_topview.npy' # Speicherort der Warping-Matrix (3x3)
RECTIFIED_TOP_VIEW_PATH = RECTIFIED_OUTPUT_DIR / 'Rectified_top_view.jpg' # Speicerort der planaren Bilder

ARUCO_DICT = cv2.aruco.DICT_6X6_250 # 6x6 ArUco Marker aus 250 verschiedenen, rotationseindeutigen Möglichkeiten

# -- Checkerboard Dimensionen --
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5 
SQUARES_X = SQUARES_VERTICALLY 
SQUARES_Y = SQUARES_HORIZONTALLY 
SQUARE_LENGTH = 0.02 # 3cm pro Square
MARKER_LENGTH = 0.01 # 1,5 cm großer Marker
LENGTH_PX = 640
MARGIN_PX = 20

#ToDo: wie werden reale physikalische Größen eingebaut


# -- Bildgröße für Debug-Ausgaben --
PREVIEW_SIZE = (840, 600)
