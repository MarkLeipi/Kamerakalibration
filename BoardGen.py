"""Erzeugt das ChArUco-Board-Bild fuer Druck oder Kontrolle."""

import cv2

from set_params import (
    BOARD_IMAGE_MARGIN_PX,
    BOARD_IMAGE_WIDTH_PX,
    BOARD_SQUARE_COUNT_X,
    BOARD_SQUARE_COUNT_Y,
    CHARUCO_BOARD_IMAGE_PATH,
    create_charuco_board,
)


def create_and_save_new_board(show_preview=True):
    """
    Erstellt aus der Projektkonfiguration ein ChArUco-Board als Bilddatei.

    Die Bildhoehe wird aus dem Seitenverhaeltnis des Boards berechnet, damit
    das erzeugte Bild zur definierten Anzahl der Quadrate passt.
    """
    board = create_charuco_board()

    board_image_height_px = int(
        round(BOARD_IMAGE_WIDTH_PX * BOARD_SQUARE_COUNT_Y / BOARD_SQUARE_COUNT_X)
    )

    board_image = cv2.aruco.CharucoBoard.generateImage(
        board,
        (BOARD_IMAGE_WIDTH_PX, board_image_height_px),
        marginSize=BOARD_IMAGE_MARGIN_PX,
    )

    if show_preview:
        cv2.imshow("ChArUco Board", board_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(str(CHARUCO_BOARD_IMAGE_PATH), board_image)
    return {
        "save_path": str(CHARUCO_BOARD_IMAGE_PATH),
        "image_shape": board_image.shape,
    }


if __name__ == "__main__":
    create_and_save_new_board()
