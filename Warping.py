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
    SQUARES_HORIZONTALLY,
    SQUARES_VERTICALLY,
    SQUARE_LENGTH,
)

def to_16_9_preview(image):
    return cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)

def show_debug_preview_windows(undistorted_image, debug_image, rectified_image):
    cv2.imshow("Undistorted", to_16_9_preview(undistorted_image))
    cv2.imshow("Detections (debug)", to_16_9_preview(debug_image))
    cv2.imshow("Rectified top-view", to_16_9_preview(rectified_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Eigentliche Funktion
def compute_topview_homography(
    image_path=str(DEFAULT_WARP_IMAGE_PATH),
    camera_matrix_path=str(CAMERA_MATRIX_PATH),
    dist_coeffs_path=str(DIST_COEFFS_PATH),
    output_image_path=str(RECTIFIED_TOP_VIEW_PATH),
    homography_path=str(HOMOGRAPHY_PATH),
    pixels_per_meter=2000.0,
    show_debug_previews=True,
):
    camera_matrix = np.load(camera_matrix_path)
    dist_coeffs = np.load(dist_coeffs_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    un_img = cv2.undistort(img, camera_matrix, dist_coeffs)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    all_board_corners = board.getChessboardCorners()

    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker_corners, marker_ids, _ = detector.detectMarkers(un_img)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers detected!")

    debug_draw = un_img.copy()
    cv2.aruco.drawDetectedMarkers(debug_draw, marker_corners, marker_ids)

    charuco_detector = cv2.aruco.CharucoDetector(board)
    detected = charuco_detector.detectBoard(un_img)
    charuco_corners, charuco_ids = detected[0], detected[1]
    retval = 0 if charuco_ids is None else len(charuco_ids)


    if charuco_ids is None or retval is None or retval < 4:
        raise RuntimeError("Not enough ChArUco corners detected!")

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
        raise RuntimeError("Fewer than 4 valid ChArUco correspondences after filtering.")

    img_pts = np.array(img_pts, dtype=np.float32)
    board_pts = np.array(board_pts, dtype=np.float32)
    board_pts_px = board_pts * pixels_per_meter

    H, inliers = cv2.findHomography(
        img_pts,
        board_pts_px,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    if inliers is not None and inliers.sum() >= 4 and inliers.sum() < len(inliers):
        img_pts_in = img_pts[inliers.ravel() == 1]
        board_pts_px_in = board_pts_px[inliers.ravel() == 1]
        H_refined, _ = cv2.findHomography(img_pts_in, board_pts_px_in, method=0)
        if H_refined is not None:
            H = H_refined

    squaresX, squaresY = board.getChessboardSize()
    square_length_m = board.getSquareLength()
    w = int(round(squaresX * square_length_m * pixels_per_meter))
    h = int(round(squaresY * square_length_m * pixels_per_meter))

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

    cv2.imwrite(output_image_path, rectified)
    np.save(homography_path, H)

    return {
        "homography": H,
        "homography_path": homography_path,
        "rectified_path": output_image_path,
        "rectified_size": (w, h),
        "charuco_corner_count": int(len(img_pts)),
    }


# Only run automatically when this file is started directly.
if __name__ == "__main__":
    result = compute_topview_homography()
    print(f"Output rectified resolution = {result['rectified_size'][0]} x {result['rectified_size'][1]} px")

