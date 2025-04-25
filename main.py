import math
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
import chess
import logging
from stockfish import Stockfish

# ---------------------------------------------------------------------------
# Suppress YOLO Logging Messages
# ---------------------------------------------------------------------------
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration and Initialization
# ---------------------------------------------------------------------------
CAMERA_ID = "../chessvid2.mp4"
WIDTH, HEIGHT = 1280, 720
YOLO_MODEL_PATH = "chess.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.6
DISPLAY_SIZE = (1280, 720)
BOARD_MARGIN = 100
CROP_OFFSET = 0  # Pixels to crop from each side after warping||change back to 30

# Initialize Stockfish (update the path as needed)
stockfish = Stockfish(
    path="C:/Users/Presision/Downloads/stockfish-windows-x86-64/stockfish/stockfish-windows-x86-64.exe"
)

COLUMNS = "abcdefgh"
ROWS = "12345678"

piece_to_fen = {
    'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B',
    'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b',
    'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k'
}

# Use a chess.Board for debugging; FENs will be built manually.
chess_board = chess.Board()
prev_board_white = chess.Board()
prev_board_black = chess.Board()
move_history = []

model = YOLO(YOLO_MODEL_PATH)
names = model.names

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


# ---------------------------------------------------------------------------
# Board Detection Functions
# ---------------------------------------------------------------------------
def find_chessboard_corners(img):
    """
    Automatically detect chessboard corners from the image.
    Returns the four corner points of the largest rectangular contour.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = rectContour(contours)
    if len(rectCon) == 0:
        return np.array([])
    biggestContour = getCornerPoints(rectCon[0])
    if biggestContour.size != 0:
        biggestContour = reorder(biggestContour)
        return biggestContour
    return np.array([])


def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon


def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left
    return myPointsNew


# ---------------------------------------------------------------------------
# Chess Analysis Functions
# ---------------------------------------------------------------------------
def warp_image(img, points, display_size=DISPLAY_SIZE, margin=BOARD_MARGIN):
    """
    Warp the chessboard to a top-down view.
    Returns the warped image, the transformation matrix, and the board size.
    """
    board_size = min(display_size) - margin
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    return img_warped, matrix, board_size


def crop_inner_squares(img_warped, board_size, offset=CROP_OFFSET):
    """
    Crop the warped image to remove the border and retain only the inner playable squares.
    Returns the cropped image and the new board size.
    """
    cropped = img_warped[offset:board_size - offset, offset:board_size - offset]
    new_board_size = board_size - 2 * offset
    return cropped, new_board_size


def draw_chess_grid(img, board_size):
    """
    Draw grid lines on the warped chessboard.
    """
    square_size = board_size // 8
    for i in range(1, 8):
        cv2.line(img, (i * square_size, 0), (i * square_size, board_size), (255, 255, 255), 2)
        cv2.line(img, (0, i * square_size), (board_size, i * square_size), (255, 255, 255), 2)
    return img


def get_chess_square(x, y, board_size):
    """
    Convert pixel coordinates (x, y) in the warped image to chess notation.
    Returns (square notation, grid indices).
    """
    square_size = board_size // 8
    grid_x = x // square_size
    grid_y = y // square_size
    if not (0 <= grid_x < 8 and 0 <= grid_y < 8):
        return "Out of Bounds", (-1, -1)
    col = COLUMNS[grid_x]
    row = ROWS[7 - grid_y]
    return f"{col}{row}", (grid_x, grid_y)


def create_fen_from_detections(piece_positions, current_turn='w'):
    """
    Convert detected pieces (and their grid positions) into a FEN string.
    The board is built as an 8x8 matrix (row 0 = top).
    Castling rights are determined dynamically by checking if kings and rooks are in their starting positions.
    """
    # Build an empty board (row 0 = top, row 7 = bottom)
    board = [['' for _ in range(8)] for _ in range(8)]
    for piece, pos in piece_positions:
        grid_x, grid_y = pos
        if 0 <= grid_x < 8 and 0 <= grid_y < 8:
            board[grid_y][grid_x] = piece_to_fen.get(piece, '')

    # Create FEN rows from the board
    fen_rows = []
    for row in board:
        empty_count = 0
        row_fen = ''
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_fen += str(empty_count)
                    empty_count = 0
                row_fen += cell
        if empty_count > 0:
            row_fen += str(empty_count)
        fen_rows.append(row_fen)
    position = '/'.join(fen_rows)

    # Dynamically determine castling rights:
    # For white, the king should be on e1 (grid position (4,7))
    # For black, the king should be on e8 (grid position (4,0))
    castling = ""
    # White castling rights:
    if board[7][4] == 'K':
        if board[7][0] == 'R':  # Rook on a1
            castling += "Q"
        if board[7][7] == 'R':  # Rook on h1
            castling += "K"
    # Black castling rights:
    if board[0][4] == 'k':
        if board[0][0] == 'r':  # Rook on a8
            castling += "q"
        if board[0][7] == 'r':  # Rook on h8
            castling += "k"
    if castling == "":
        castling = "-"

    return f"{position} {current_turn} {castling} - 0 1"


def detect_pieces(img_warped, board_size):
    """
    Detect chess pieces on the warped chessboard using YOLO.
    Returns a list of detected pieces with their grid positions and draws the detections.
    """
    detected_pieces = []
    results = model(img_warped)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_id = int(box.cls[0])
            class_name = names[class_id]
            conf = float(box.conf[0])
            if conf < DETECTION_CONFIDENCE_THRESHOLD:
                continue
            square, (grid_x, grid_y) = get_chess_square(cx, cy, board_size)
            if grid_x == -1 or grid_y == -1:
                continue
            detected_pieces.append((class_name, (grid_x, grid_y)))
            cvzone.putTextRect(
                img_warped,
                f'{piece_to_fen[class_name]} {square}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
                colorR=(255, 255, 0)
            )
    return detected_pieces


def square_to_pixel(square, board_size):
    """
    Convert a square in algebraic notation (e.g., "e4") to pixel coordinates (center)
    on the warped chessboard image.
    """
    square_size = board_size / 8
    col = COLUMNS.index(square[0])
    row = 8 - int(square[1])
    x = int(col * square_size + square_size / 2)
    y = int(row * square_size + square_size / 2)
    return (x, y)


# ---------------------------------------------------------------------------
# Main Loop with Automatic Board Detection, Cropping, and Rotation
# ---------------------------------------------------------------------------
def main():
    last_stable_fen = None  # Last confirmed FEN (after stability check)
    stable_fen = None  # FEN candidate that is being confirmed
    fen_counter = 0
    STABILITY_THRESHOLD = 5  # Frames required for stability
    best_moves_white = []  # Top moves for White
    best_moves_black = []  # Top moves for Black

    # Board detection mode flag
    board_detection_mode = True
    board_corners = None

    # Rotation state: 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
    rotation_state = 0

    global prev_board_white,prev_board_black, move_history


    while True:
        success, img = cap.read()
        if not success:
            break

        img_display = img.copy()
        imgBlank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # Blank image for debugging

        if board_detection_mode:
            # Detect chessboard corners automatically
            corners = find_chessboard_corners(img)
            if corners.size != 0:
                cv2.drawContours(img_display, [corners], -1, (0, 255, 0), 10)
                cvzone.putTextRect(
                    img_display,
                    "Chessboard detected! Press 'C' to confirm or 'R' to retry",
                    (50, 50),
                    scale=1,
                    thickness=2,
                    colorR=(0, 255, 0)
                )
                cv2.imshow("Chessboard Detection", img_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    board_corners = corners
                    board_detection_mode = False
                    rotation_state = 0  # Reset rotation
                    cv2.destroyWindow("Chessboard Detection")
                elif key == ord('q'):
                    break
            else:
                cvzone.putTextRect(
                    img_display,
                    "No chessboard detected. Adjust camera position.",
                    (50, 50),
                    scale=1,
                    thickness=2,
                    colorR=(0, 0, 255)
                )
                cv2.imshow("Chessboard Detection", img_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # Warp the image using detected board corners
            img_warped, matrix, board_size = warp_image(img, board_corners.reshape(4, 2))
            # Apply rotation if needed
            if rotation_state == 1:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_state == 2:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_180)
            elif rotation_state == 3:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Crop out the extra border so only the inner squares remain
            img_warped_cropped, new_board_size = crop_inner_squares(img_warped, board_size, offset=CROP_OFFSET)

            # Draw grid lines on the cropped image and detect pieces there
            img_warped_cropped = draw_chess_grid(img_warped_cropped, new_board_size)
            detected_pieces = detect_pieces(img_warped_cropped, new_board_size)

            # Create a FEN candidate from detections
            current_fen_candidate = create_fen_from_detections(detected_pieces, current_turn='w')

            # turn_char = 'w' if prev_board.turn == chess.WHITE else 'b'
            # current_fen_candidate = create_fen_from_detections(detected_pieces, current_turn=turn_char)

            # Stability check for FEN detection over consecutive frames
            if current_fen_candidate == stable_fen:
                fen_counter += 1
            else:
                stable_fen = current_fen_candidate
                fen_counter = 1

            if fen_counter >= STABILITY_THRESHOLD and stable_fen != last_stable_fen:
                last_stable_fen = stable_fen

                # try:
                #     new_board = chess.Board(stable_fen)
                #     # find actual move from prev_board
                #     played = None
                #     for m in prev_board.legal_moves:
                #         tb = prev_board.copy()
                #         tb.push(m)
                #         if tb.board_fen() == new_board.board_fen():
                #             played = m
                #             break
                #     if played:
                #         san = prev_board.san(played)
                #         turn = 'White' if prev_board.turn == chess.WHITE else 'Black'
                #         stockfish.set_fen_position(stable_fen)
                #         eval_info = stockfish.get_evaluation()
                #         cp = eval_info.get('value')
                #         move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                #     prev_board = new_board
                #     chess_board.set_fen(stable_fen)
                #     last_stable_fen = stable_fen
                #     print("Move History:")
                #     for h in move_history:
                #         print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")
                # except Exception as e:
                #     print("History error:", e)

                # try:
                #
                #     stockfish.set_fen_position(stable_fen)
                #     new_board = chess.Board(stockfish.get_fen_position())
                #     played = None
                #     for m in prev_board.legal_moves:
                #         tb = prev_board.copy()
                #         tb.push(m)
                #         if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                #             played = m
                #             print(f"debugg {tb.board_fen().split(' ')[0]} {new_board.board_fen().split(' ')[0]}played: {played}")
                #             break
                #     if played:
                #         san = prev_board.san(played)
                #         turn = 'White' if prev_board.turn == chess.WHITE else 'Black'
                #         # stockfish.set_fen_position(stable_fen)
                #         eval_info = stockfish.get_evaluation()
                #         cp = eval_info.get('value')
                #         move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                #     prev_board = new_board
                #     chess_board.set_fen(stockfish.set_fen_position())
                # except Exception as e:
                #     print("History update error:", e)


                try:
                    temp_board = chess.Board(stable_fen)
                    if temp_board.is_valid():
                        chess_board.set_fen(stable_fen)
                        print("\nUpdated Chess Board:")
                        print(chess_board)
                        print("Stable FEN:", stable_fen)
                        fen_white = stable_fen.split(' ')[0] + " w KQkq - 0 1"
                        fen_black = stable_fen.split(' ')[0] + " b KQkq - 0 1"
                        try:

                            stockfish.set_fen_position(fen_white)
                            best_moves_white = stockfish.get_top_moves(3)

                            new_board = chess.Board(fen_white)
                            # find actual move from prev_board
                            played = None
                            for m in prev_board_white.pseudo_legal_moves:
                                tb = prev_board_white.copy()
                                tb.push(m)
                                if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                    played = m
                                    break
                            if played:
                                san = prev_board_white.san(played)
                                turn = 'White' if prev_board_white.turn == chess.WHITE else 'Black'
                                stockfish.set_fen_position(fen_white)
                                eval_info = stockfish.get_evaluation()
                                cp = eval_info.get('value')
                                move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                            prev_board_white = new_board
                            chess_board.set_fen(fen_white)
                            last_stable_fen = stable_fen
                            # print("Move History:")
                            # for h in move_history:
                            #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")

                        except Exception as e:
                            print("Error getting top moves for White:", e)
                            best_moves_white = []
                        try:
                            stockfish.set_fen_position(fen_black)
                            best_moves_black = stockfish.get_top_moves(3)

                            new_board = chess.Board(fen_black)
                            # find actual move from prev_board_black
                            played = None
                            for m in prev_board_black.pseudo_legal_moves:
                                tb = prev_board_black.copy()
                                tb.push(m)
                                if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                    played = m
                                    break
                            if played:
                                san = prev_board_black.san(played)
                                turn = 'White' if prev_board_black.turn == chess.WHITE else 'Black'
                                stockfish.set_fen_position(fen_black)
                                eval_info = stockfish.get_evaluation()
                                cp = eval_info.get('value')
                                move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                            prev_board_black = new_board
                            chess_board.set_fen(fen_black)
                            last_stable_fen = stable_fen
                            # print("Move History:")
                            # for h in move_history:
                            #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")
                        except Exception as e:
                            print("Error getting top moves for Black:", e)
                            best_moves_black = []
                        print("Top moves for White:", best_moves_white)
                        print("Top moves for Black:", best_moves_black)
                    else:
                        print("Detected FEN is not valid:", stable_fen)
                        best_moves_white = best_moves_black = []
                except Exception as e:
                    print("Error processing FEN:", stable_fen, e)
                    best_moves_white = best_moves_black = []

                # print move history
                print("Move History:")
                for h in move_history:
                    print(f"{h['turn']} {h['move']} ⇒ {h['evaluation_cp']} cp")

            y0 = 30
            dy = 30
            for i, h in enumerate(move_history):  # show last 10 moves
                text = f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp"
                cv2.putText(
                    imgBlank,
                    text,
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            # Draw arrows for top moves on the cropped image
            for move_info in best_moves_white:
                move = move_info.get("Move", None)
                score = move_info.get("Centipawn", None)

                if move and len(move) >= 4:
                    start_square = move[:2]
                    end_square = move[2:4]
                    start_px = square_to_pixel(start_square, new_board_size)
                    end_px = square_to_pixel(end_square, new_board_size)
                    cv2.arrowedLine(img_warped_cropped, start_px, end_px, (255, 0, 0), 2, tipLength=0.3)
                    cv2.putText(img_warped_cropped, f"{score}", (end_px[0], end_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for move_info in best_moves_black:
                move = move_info.get("Move", None)
                score = move_info.get("Centipawn", None)

                if move and len(move) >= 4:
                    start_square = move[:2]
                    end_square = move[2:4]
                    start_px = square_to_pixel(start_square, new_board_size)
                    end_px = square_to_pixel(end_square, new_board_size)
                    cv2.arrowedLine(img_warped_cropped, start_px, end_px, (0, 0, 255), 2, tipLength=0.3)
                    cv2.putText(img_warped_cropped, f"{score}", (end_px[0], end_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Display best moves as text on the original image
            display_text = ""
            if best_moves_white:
                white_moves = ", ".join([m.get("Move", "") for m in best_moves_white])
                display_text += f"White: {white_moves}  "
            if best_moves_black:
                black_moves = ", ".join([m.get("Move", "") for m in best_moves_black])
                display_text += f"Black: {black_moves}"
            if display_text:
                cv2.putText(img, display_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display rotation and recalibration instructions
            rotation_text = f"Current Rotation: {rotation_state * 90}°"
            cvzone.putTextRect(
                img_warped_cropped,
                rotation_text,
                (10, 30),
                scale=1,
                thickness=2,
                colorR=(0, 0, 0)
            )
            cvzone.putTextRect(
                img,
                "Press 'R' to rotate board | 'D' to detect new board",
                (50, HEIGHT - 50),
                scale=1,
                thickness=2,
                colorR=(0, 0, 0)
            )
            # cv2.imshow("Warped Chessboard", img_warped_cropped)
            # cv2.imshow("Original Image", img)
            # played_moves = [ f"{h['turn']} {h['move']} ⇒ {h['evaluation_cp']} cp" for h in move_history]
            # cv2.putText(imgBlank,f"{played_moves}"+"\n",(0,0),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



            cvzone.putTextRect(
                imgBlank,
                "Move History",
                (700, 50),
                scale=5,
                thickness=2,
                colorR=(0, 255, 0)
            )

            imgStack = cvzone.stackImages([img_warped_cropped,imgBlank,img],2,0.7)
            cv2.imshow("Stacked Image", imgStack)

            # imgStack = cvzone.stackImages([img_warped_cropped, imgBlank, img], 2, 1.5)
            #
            # # Create a resizable named window
            # cv2.namedWindow("Stacked Image", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Stacked Image", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
            #
            # cv2.imshow("Stacked Image", imgStack)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Rotate the warped image (90° per press)
                rotation_state = (rotation_state + 1) % 4
                # Reset FEN stability on rotation change
                fen_counter = 0
                last_stable_fen = None
                stable_fen = None
                best_moves_white = []
                best_moves_black = []
            elif key == ord('d'):
                # Switch back to board detection mode
                board_detection_mode = True
                # cv2.destroyWindow("Warped Chessboard")
                # cv2.destroyWindow("Original Image")
                cv2.destroyWindow("Stacked Image")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
