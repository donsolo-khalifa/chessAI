import cv2
import numpy as np
import pickle
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Webcam settings
cam_id = 1  # Change to 1 if needed
width, height = 1280, 720

# Load predefined chessboard corners
board_corners_file = "chessboard_corners.p"
with open(board_corners_file, 'rb') as f:
    board_corners = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(cam_id)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Chessboard notation
columns = "abcdefgh"
rows = "12345678"


def warp_image(img, points):
    """Warp the detected chessboard to a top-down view."""
    display_width, display_height = 1280, 720
    board_size = min(display_width, display_height) - 100  # Fit within the screen

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(img, matrix, (int(board_size), int(board_size)))

    return imgWarped, matrix, int(board_size)


# def get_chess_square(x, y, board_size):
#     """Map pixel positions to chess notation."""
#     square_size = board_size // 8
#     if not (0 <= x < board_size) or not (0 <= y < board_size):
#         return "Out of Bounds"
#     col = columns[x // square_size]
#     row = rows[7 - (y // square_size)]
#     return f"{col}{row}", (x // square_size, y // square_size)

# works
# def get_chess_square(x, y, board_size):
#     """Map pixel positions to chess notation."""
#     square_size = board_size // 8
#     if not (0 <= x < board_size) or not (0 <= y < board_size):
#         return "Out of Bounds", (-1, -1)  # Ensure it returns two values
#
#     col = columns[x // square_size]
#     row = rows[7 - (y // square_size)]
#     return f"{col}{row}", (x // square_size, y // square_size)

# this is inverting
# def get_chess_square(x, y, board_size):
#     """Map pixel positions to chess notation."""
#     square_size = board_size // 8
#     grid_x = x // square_size
#     grid_y = 7 - (y // square_size)  # Flip row for chess notation
#
#     # ✅ Ensure indexes are within valid range (0-7)
#     if not (0 <= grid_x < 8) or not (0 <= grid_y < 8):
#         return "Out of Bounds", (-1, -1)  # Prevent IndexError
#
#     col = columns[grid_x]
#     row = rows[grid_y]
#     return f"{col}{row}", (grid_x, grid_y)

def get_chess_square(x, y, board_size):
    """Map pixel positions to chess notation."""
    square_size = board_size // 8
    grid_x = x // square_size
    grid_y = y // square_size  # ✅ Fix: Do not flip the row here

    # ✅ Ensure indexes are within valid range (0-7)
    if not (0 <= grid_x < 8) or not (0 <= grid_y < 8):
        return "Out of Bounds", (-1, -1)  # Prevent IndexError

    col = columns[grid_x]
    row = rows[7 - grid_y]  # ✅ Flip notation here for chess coordinate system
    return f"{col}{row}", (grid_x, grid_y)


def draw_chess_grid(img, board_size):
    """Draws grid lines on the warped chessboard."""
    square_size = board_size // 8
    for i in range(1, 8):
        cv2.line(img, (i * square_size, 0), (i * square_size, board_size), (255, 255, 255), 2)
        cv2.line(img, (0, i * square_size), (board_size, i * square_size), (255, 255, 255), 2)
    return img


while True:
    success, img = cap.read()
    if not success:
        break

    # Warp the chessboard
    imgWarped, matrix, board_size = warp_image(img, board_corners)
    imgWarped = draw_chess_grid(imgWarped, board_size)  # Draw grid lines

    # Detect hands
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        index_finger = hand["lmList"][8][:2]

        # Transform fingertip position to board coordinates
        point = np.array([[index_finger]], dtype=np.float32)
        warped_point = cv2.perspectiveTransform(point, matrix)[0][0]

        # Get chess square
        square, (grid_x, grid_y) = get_chess_square(int(warped_point[0]), int(warped_point[1]), board_size)

        if grid_x != -1 and grid_y != -1:  # Ensure it's within bounds
            # Highlight the selected square in cyan
            square_size = board_size // 8
            top_left = (grid_x * square_size, grid_y * square_size)
            bottom_right = ((grid_x + 1) * square_size, (grid_y + 1) * square_size)
            cv2.rectangle(imgWarped, top_left, bottom_right, (255, 255, 0), -1)  # Cyan fill

            # Redraw grid lines on top of the highlight
            imgWarped = draw_chess_grid(imgWarped, board_size)

        cv2.putText(imgWarped, square, (int(warped_point[0]), int(warped_point[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display images
    # cv2.namedWindow("Warped Chessboard", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped Chessboard", imgWarped)
    cv2.imshow("Original Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
