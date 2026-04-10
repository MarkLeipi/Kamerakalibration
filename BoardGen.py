import cv2
from set_params import (
    ARUCO_DICT,
    CHARUCO_BOARD_IMAGE_PATH,
    LENGTH_PX,
    MARGIN_PX,
    MARKER_LENGTH,
    SQUARE_LENGTH,
    SQUARES_HORIZONTALLY,
    SQUARES_VERTICALLY,
)

def create_and_save_new_board(show_preview=True):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)

    if show_preview:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(str(CHARUCO_BOARD_IMAGE_PATH), img)
    return {
        "save_path": str(CHARUCO_BOARD_IMAGE_PATH),
        "image_shape": img.shape,
    }


# Only run automatically when this file is started directly.
if __name__ == "__main__":
    create_and_save_new_board()
