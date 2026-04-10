########################################################################
# run_calibration_and_warping.py
# Startet nacheinander:
#   1. Kamerakalibrierung  (Calibration.py)
#   2. Warping / Draufsicht-Berechnung  (Warping.py)
#
# Parameter:
#   show_detection_previews=True  → zeigt kurz die erkannten Board-Ecken
#   show_undistorted_previews=True → zeigt die entzerrten Kalibrierbilder
#   show_debug_previews=True      → zeigt Zwischenergebnisse des Warpings
########################################################################
from Calibration import calibrate_and_save_parameters
from Warping import compute_topview_homography


def main():
    print("=== SCHRITT 1: Kamera-Kalibrierung ===")

    calib_result = calibrate_and_save_parameters(
        show_detection_previews=True,
        show_undistorted_previews=True,
    )

    print("Kalibrierung abgeschlossen:")
    print(f"  Bilder gesamt:        {calib_result['image_count']}")
    print(f"  Detektierte Frames:   {calib_result['detected_frame_count']}")
    print(f"  RMS-Fehler:           {calib_result['rms']:.4f}")

    print("\n=== SCHRITT 2: Warping / Draufsicht ===")

    warp_result = compute_topview_homography(
        show_debug_previews=True,
    )

    print("Warping abgeschlossen:")
    print(f"  Ausgabeauflösung: {warp_result['rectified_size'][0]} x {warp_result['rectified_size'][1]} px")
    print(f"  Erkannte Ecken:   {warp_result['charuco_corner_count']}")


if __name__ == "__main__":
    main()
