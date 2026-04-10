########################################################################
# run_calibration_and_warping
# Führt Kamerakalibration und Warping aus

# INPUTS der importierten Funktionen:
# show_debug: True --> Zeigt die detektierten Ecken des Checkerboards in 150ms Abstand
# show_undistorted: True --> Zeigt die entzerrten Kalibrierbilder in 150ms Abstand
# show_debug_previews: True --> 
########################################################################
from Calibration import calibrate_and_save_parameters
from Warping import compute_topview_homography


def main():
    print("=== STEP 1: Kamera-Kalibrierung ===")

    calib_result = calibrate_and_save_parameters(
        show_debug=True,
        show_undistorted=True,
    )

    print("Kalibrierung abgeschlossen:")
    print(f"  Bilder gesamt:        {calib_result['image_count']}")
    print(f"  Detektierte Frames:   {calib_result['detected_frame_count']}")
    print(f"  RMS-Fehler:           {calib_result['rms']:.4f}")

    print("\n=== STEP 2: Warping / Top-View ===")

    warp_result = compute_topview_homography(
        show_debug_previews=True,
    )

    print("Warping abgeschlossen:")
    print(f"  Rectified size: {warp_result['rectified_size'][0]} x {warp_result['rectified_size'][1]} px")
    print(f"  Erkannte Ecken: {warp_result['charuco_corner_count']}")


if __name__ == "__main__":
    main()
