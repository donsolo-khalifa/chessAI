# Chess AI Vision System

This project integrates computer vision with chess engine analysis to create a real-time chess board analyzer. Using YOLO for chess piece detection and Stockfish for move analysis, the system generates a live FEN (Forsyth–Edwards Notation) string from a webcam feed and displays the top three move suggestions for both White and Black by drawing colored arrows on the board.

In the latest update, the chessboard is detected automatically from the webcam feed—eliminating the need for manual corner selection. The system now warps the board based on the automatically detected corners, then crops and rotates the image as needed.

> **Note:** I used the Iriun Webcam App on both my phone and pc to use my phone as a camera, so the setup works with both built-in webcams and external devices.

## Features

- **Automatic Chessboard Detection**
  The updated script now automatically detects the chessboard by locating the largest rectangular contour in the webcam feed. No manual calibration is required.
  
- **Real-Time Chess Piece Detection**  
  Uses a YOLO model to detect chess pieces from a live webcam feed.

- **FEN Generation & Stability Check**  
  Converts piece detections into a FEN string with a stability filter to reduce jitter.
  
- **Stockfish Integration**  
  Queries Stockfish for the top three move suggestions for both White (blue arrows) and Black (red arrows).
  
- **Visual Feedback**  
  Draws a chess grid, arrows, and Centipawn score representing the best moves on a warped, top-down view of the chessboard.
  
- **Optional Manual Calibration**  
  The project still includes a calibration script (`edgeSelect.py`)  for cases where automatic detection might not work perfectly, but it is no longer the primary method.
  A window will open showing the webcam feed.
  Click four points corresponding to the corners of the chessboard (top-left, top-right, bottom-left, bottom-right) in order.
  Press `s` to save the corners.
  The saved file (chessboard_corners.p) will be used for warping the board.
  > To be used for `unMained.py`
  
## Installation

### Prerequisites

- Python 3.7 or higher
- A working external camera (e.g., via the Iriun Webcam App to use your phone as a camera) [Iriun website](https://iriun.com/)
- Make sure your smart phone and pc are connected to the same wifi network when running Iriun on your pc and phone
- Stockfish chess engine (download from [Stockfish website](https://stockfishchess.org/download/))
- YOLO chess piece model `chess.pt` (download from [Google Drive](https://drive.google.com/drive/folders/1U2TNd5IYTdHvQC1QEWKOrykBnaUSf1Lv?usp=sharing))
- clone the repository
  ```bash
  git clone https://github.com/donsolo-khalifa/chessAI.git
  cd chessAI
  ```

### Required Python Packages

Install the required packages using pip:

```bash
pip install -r requirements.txt
```
### Usage
- **1. Configure your environment**
Copy `.env.example` to `.env` and set the values for your machine:
```bash
cp .env.example .env
```
- `STOCKFISH_PATH`: full path to your Stockfish executable (optional if `stockfish` is on your `PATH`)
- `YOLO_MODEL_PATH`: path to the YOLO weights (default `chess.pt`)
- `CAMERA_ID`: webcam index (`0` built-in, `1` second camera such as Iriun) or a video file path

Download the YOLO weights file `chess.pt` from [Google Drive](https://drive.google.com/drive/folders/1U2TNd5IYTdHvQC1QEWKOrykBnaUSf1Lv?usp=sharing) and place it in the project root, then run:
```bash
python main.py
```
The application will start capturing the webcam feed.
The chessboard image is warped into a top-down view, and a grid is drawn.
YOLO detects chess pieces on the board and converts the detections into a FEN string.
Stockfish analyzes the board and the top three moves for both White (displayed as blue arrows) and Black (displayed as red arrows) are drawn on the warped image.
The original webcam feed and the warped chessboard are displayed alongside each other.

- **2. Interacting with App**
> If needed
- Rotate the Board: Press `r` to rotate the board view by 90° increments if warpped chess board is not upright.
- Recalibrate the Board: Press `d` to re-initiate automatic board detection if the chessboard is not detected correctly.
- Quit: Press `q` to exit the application.


