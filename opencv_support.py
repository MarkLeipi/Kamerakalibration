"""Zentrale OpenCV-/ChArUco-Kompatibilitaet fuer dieses Projekt."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import cv2


def _major_version(version_text: str) -> Optional[int]:
    """Liest die Hauptversion aus einem OpenCV-Versionsstring."""
    try:
        return int(version_text.split(".")[0])
    except (ValueError, AttributeError, IndexError):
        return None


def _create_detector_parameters():
    """Erzeugt Detector-Parameter fuer alte und neue OpenCV-APIs."""
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise RuntimeError(
        "OpenCV kann keine ArUco-Detektorparameter erzeugen. "
        "Bitte installiere opencv-contrib-python 4.x."
    )


def _normalize_interpolate_result(raw_result) -> Tuple[int, Any, Any]:
    """Normalisiert die Rueckgabe von interpolateCornersCharuco."""
    if not isinstance(raw_result, tuple) or len(raw_result) < 3:
        return 0, None, None

    retval, corners, ids = raw_result[:3]
    count = 0 if retval is None else int(retval)
    return count, corners, ids


def _normalize_detector_result(raw_result) -> Tuple[int, Any, Any]:
    """Normalisiert die Rueckgabe von CharucoDetector.detectBoard."""
    if not isinstance(raw_result, tuple) or len(raw_result) < 2:
        return 0, None, None

    corners, ids = raw_result[:2]
    count = 0 if ids is None else len(ids)
    return count, corners, ids


@dataclass(frozen=True)
class CharucoRuntime:
    """Merkt sich einmalig, welche OpenCV-API im Projekt genutzt wird."""

    version: str
    marker_backend_name: str
    charuco_backend_name: str
    calibration_mode: str
    _detect_markers_fn: Callable[[Any], Tuple[Any, Any, Any]]
    _detect_charuco_fn: Callable[[Any, Any, Any], Tuple[int, Any, Any]]

    @property
    def description(self) -> str:
        return (
            f"OpenCV {self.version} mit {self.marker_backend_name} "
            f"und {self.charuco_backend_name}"
        )

    def detect_markers(self, image):
        return self._detect_markers_fn(image)

    def detect_charuco_corners(self, image, marker_corners, marker_ids):
        if marker_ids is None or len(marker_ids) == 0:
            return 0, None, None
        return self._detect_charuco_fn(image, marker_corners, marker_ids)

    def draw_charuco_corners(self, image, charuco_corners, charuco_ids):
        if (
            charuco_corners is not None
            and charuco_ids is not None
            and hasattr(cv2.aruco, "drawDetectedCornersCharuco")
        ):
            cv2.aruco.drawDetectedCornersCharuco(
                image,
                charuco_corners,
                charuco_ids,
                (0, 255, 0),
            )


def create_charuco_runtime(board, dictionary) -> CharucoRuntime:
    """Waehlt genau einmal die passende OpenCV-Strategie fuer dieses Projekt."""
    version = getattr(cv2, "__version__", "unbekannt")
    major_version = _major_version(version)

    if major_version is not None and major_version < 4:
        raise RuntimeError(
            f"Gefundene OpenCV-Version: {version}. "
            "Dieses Projekt benoetigt OpenCV 4.x mit dem aruco-Modul "
            "(meist aus opencv-contrib-python)."
        )

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "In dieser OpenCV-Installation fehlt das aruco-Modul. "
            "Bitte installiere opencv-contrib-python 4.x."
        )

    if not hasattr(board, "matchImagePoints"):
        raise RuntimeError(
            "Das geladene OpenCV unterstuetzt board.matchImagePoints nicht. "
            "Diese Funktion wird fuer Kalibrierung und Warping benoetigt."
        )

    detector_parameters = _create_detector_parameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        marker_detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)
        detect_markers_fn = marker_detector.detectMarkers
        marker_backend_name = "ArucoDetector"
    elif hasattr(cv2.aruco, "detectMarkers"):
        def detect_markers_fn(image):
            return cv2.aruco.detectMarkers(
                image,
                dictionary,
                parameters=detector_parameters,
            )

        marker_backend_name = "detectMarkers"
    else:
        raise RuntimeError(
            "In dieser OpenCV-Installation fehlt eine nutzbare ArUco-Markererkennung."
        )

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        def detect_charuco_fn(image, marker_corners, marker_ids):
            return _normalize_interpolate_result(
                cv2.aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    image,
                    board,
                )
            )

        charuco_backend_name = "interpolateCornersCharuco"
    elif hasattr(cv2.aruco, "CharucoDetector"):
        charuco_detector = cv2.aruco.CharucoDetector(board)

        def detect_charuco_fn(image, marker_corners, marker_ids):
            _ = marker_corners, marker_ids
            return _normalize_detector_result(charuco_detector.detectBoard(image))

        charuco_backend_name = "CharucoDetector"
    else:
        raise RuntimeError(
            "Es wurde keine passende ChArUco-Erkennung gefunden. "
            "Unterstuetzt werden OpenCV-Installationen mit "
            "interpolateCornersCharuco oder CharucoDetector."
        )

    calibration_mode = (
        "charuco"
        if hasattr(cv2.aruco, "calibrateCameraCharuco")
        else "standard"
    )

    return CharucoRuntime(
        version=version,
        marker_backend_name=marker_backend_name,
        charuco_backend_name=charuco_backend_name,
        calibration_mode=calibration_mode,
        _detect_markers_fn=detect_markers_fn,
        _detect_charuco_fn=detect_charuco_fn,
    )
