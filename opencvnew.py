from pathlib import Path
import cv2
import numpy as np

# --- PFADE & PARAMETER ---
BASE_DIR = Path(__file__).resolve().parent
CAL_IMAGES_DIR = BASE_DIR / "saved_frames_Iphone"
WARP_IMAGES_DIR = BASE_DIR / "Warp_Images"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Pfade für Ergebnisse
CAMERA_MATRIX_PATH = OUTPUTS_DIR / "calibration" / "camera_matrix.npy"
DIST_COEFFS_PATH = OUTPUTS_DIR / "calibration" / "dist_coeffs.npy"
HOMOGRAPHY_PATH = OUTPUTS_DIR / "homography" / "H_charuco_topview.npy"
RECTIFIED_TOP_VIEW_PATH = OUTPUTS_DIR / "rectified" / "Rectified_top_view.jpg"
CALIBRATION_DEBUG_OUTPUT_DIR = OUTPUTS_DIR / "calibration_debug"

# Board-Definition (5x7, 20mm Quadrate, 10mm Marker)
BOARD_DIM = (5, 7)
SQUARE_LEN = 0.02
MARKER_LEN = 0.01
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Debug & Anzeige
PREVIEW_SIZE = (800, 600)
SAVE_DEBUG = True
NMB_DEBUG = 5
DEFAULT_WARP_IMAGE_PATH = WARP_IMAGES_DIR / "5x7vid.jpg"

# --- FUNKTIONEN ---

def create_charuco_board():
    return cv2.aruco.CharucoBoard(BOARD_DIM, SQUARE_LEN, MARKER_LEN, ARUCO_DICT)

def create_charuco_runtime(board=None, dictionary=None):
    """Kapselt den Detector für Kalibrierung und Warping."""
    bd = board if board else create_charuco_board()
    dic = dictionary if dictionary else ARUCO_DICT
    
    # Detector-Parameter (standardmäßig robust)
    params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(bd, charuco_params, params)
    
    # Simuliert die 'runtime' Struktur aus deinem Code
    class Runtime:
        def __init__(self, det):
            self.det = det
            self.description = "ChArUco Detector (v2 reduced)"
            self.calibration_mode = "charuco"
        def detect_markers(self, img): return self.det.getDetectorParameters(), self.det.getDictionary(), None
        def detect_charuco_corners(self, img, corners, ids): 
            return self.det.detectBoard(img)
        def draw_charuco_corners(self, img, corners, ids): 
            cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
            
    return Runtime(detector)

def ensure_output_directories():
    for d in [OUTPUTS_DIR / s for s in ["calibration", "calibration_debug", "homography", "rectified"]]:
        d.mkdir(parents=True, exist_ok=True)

def create_charuco_dictionary(): 
    return ARUCO_DICT
