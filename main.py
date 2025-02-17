import math
import cv2
import numpy as np
import pickle
import cvzone
from ultralytics import YOLO
import chess
import logging
from stockfish import Stockfish  # New import

# -----------------------------------------------------------------------------
# Suppress YOLO Logging Messages
# -----------------------------------------------------------------------------
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------
CAMERA_ID = 1
WIDTH, HEIGHT = 1280, 720
YOLO_MODEL_PATH = "chess.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.6
DISPLAY_SIZE = (1280, 720)
BOARD_MARGIN = 100

# Initialize Stockfish (update path to your Stockfish executable)
stockfish = Stockfish(path="C:/Users/Presision/Downloads/stockfish-windows-x86-64/stockfish/stockfish-windows-x86-64.exe")

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
model = YOLO(YOLO_MODEL_PATH)
names = model.names

with open("chessboard_corners.p", 'rb') as f:
    board_corners = pickle.load(f)

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
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
    """
    board = [['' for _ in range(8)] for _ in range(8)]
    for piece, pos in piece_positions:
        grid_x, grid_y = pos
        if 0 <= grid_x < 8 and 0 <= grid_y < 8:
            board[grid_y][grid_x] = piece_to_fen.get(piece, '')

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
    # Standard castling rights and no en passant for simplicity
    return f"{position} {current_turn} KQkq - 0 1"

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
    # In our warped board, rank 8 is at the top (row 0) and rank 1 is at the bottom (row 7)
    row = 8 - int(square[1])
    x = int(col * square_size + square_size / 2)
    y = int(row * square_size + square_size / 2)
    return (x, y)

# -----------------------------------------------------------------------------
# Main Loop with Stability Check, Validity Check, Dual Move Analysis,
# and Drawing Arrows for Top 3 Moves for White and Black.
# -----------------------------------------------------------------------------
def main():
    last_stable_fen = None  # Last confirmed FEN (after stability check)
    stable_fen = None       # FEN candidate that is being confirmed
    fen_counter = 0
    STABILITY_THRESHOLD = 5  # Number of consecutive frames required to accept the detection
    best_moves_white = []  # List to store top moves for White
    best_moves_black = []  # List to store top moves for Black

    while True:
        success, img = cap.read()
        if not success:
            break

        # Warp the image and detect pieces
        img_warped, matrix, board_size = warp_image(img, board_corners)
        img_warped = draw_chess_grid(img_warped, board_size)
        detected_pieces = detect_pieces(img_warped, board_size)
        # Create FEN candidate with default turn 'w'
        current_fen_candidate = create_fen_from_detections(detected_pieces, current_turn='w')

        # Stability check to ensure FEN is consistent over several frames
        if current_fen_candidate == stable_fen:
            fen_counter += 1
        else:
            stable_fen = current_fen_candidate
            fen_counter = 1

        # Once the FEN is stable for enough frames, update it
        if fen_counter >= STABILITY_THRESHOLD and stable_fen != last_stable_fen:
            last_stable_fen = stable_fen

            try:
                # Validate the FEN using python-chess.
                temp_board = chess.Board(stable_fen)
                if temp_board.is_valid():
                    # Update chess_board for debugging
                    chess_board.set_fen(stable_fen)
                    print("\nUpdated Chess Board:")
                    print(chess_board)
                    print("Stable FEN:", stable_fen)

                    # Create two FENs – one with White to move, one with Black.
                    fen_white = stable_fen.split(' ')[0] + " w KQkq - 0 1"
                    fen_black = stable_fen.split(' ')[0] + " b KQkq - 0 1"

                    # Get top 3 moves for White
                    try:
                        stockfish.set_fen_position(fen_white)
                        best_moves_white = stockfish.get_top_moves(3)
                    except Exception as e:
                        print("Error getting top moves for White:", e)
                        best_moves_white = []
                    # Get top 3 moves for Black
                    try:
                        stockfish.set_fen_position(fen_black)
                        best_moves_black = stockfish.get_top_moves(3)
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

        # Draw arrows for top moves on the warped image
        # Use blue arrows for White and red arrows for Black
        for move_info in best_moves_white:
            move = move_info.get("Move", None)
            if move and len(move) >= 4:
                start_square = move[:2]
                end_square = move[2:4]
                start_px = square_to_pixel(start_square, board_size)
                end_px = square_to_pixel(end_square, board_size)
                # Draw arrow: blue color (BGR: 255, 0, 0), thickness 2
                cv2.arrowedLine(img_warped, start_px, end_px, (255, 0, 0), 2, tipLength=0.3)
        for move_info in best_moves_black:
            move = move_info.get("Move", None)
            if move and len(move) >= 4:
                start_square = move[:2]
                end_square = move[2:4]
                start_px = square_to_pixel(start_square, board_size)
                end_px = square_to_pixel(end_square, board_size)
                # Draw arrow: red color (BGR: 0, 0, 255), thickness 2
                cv2.arrowedLine(img_warped, start_px, end_px, (0, 0, 255), 2, tipLength=0.3)

        # Display best moves as text on the original image if available
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

        cv2.imshow("Warped Chessboard", img_warped)
        cv2.imshow("Original Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
