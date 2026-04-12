

from Calibration import calibrate_and_save_parameters
from Warping import compute_topview_homography


def main():
    print("=== STEP 1: Kamera-Kalibrierung ===")

    calibration_result = calibrate_and_save_parameters(
        save_debug_images=True,
    )

    print("Kalibrierung abgeschlossen:")
    print(f"  Bilder gesamt:        {calibration_result['image_count']}")
    print(f"  Erfolgreich ausgewertete Bilder:   {calibration_result['detected_frame_count']}")
    print(f"  RMS-Fehler:           {calibration_result['rms']:.4f}")

    print("\n=== STEP 2: Warping / Top-View ===")

    warp_result = compute_topview_homography(show_debug_previews=True)

    print("Warping abgeschlossen:")
    print(
        "  Rectified size: "
        f"{warp_result['rectified_size'][0]} x {warp_result['rectified_size'][1]} px"
    )
    print(f"  Erkannte Ecken: {warp_result['charuco_corner_count']}")


if __name__ == "__main__":
    main()
