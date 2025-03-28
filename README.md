# Chess AI Vision System

This project integrates computer vision with chess engine analysis to create a real-time chess board analyzer. Using YOLO for chess piece detection and Stockfish for move analysis, the system generates a live FEN (Forsyth–Edwards Notation) string from a webcam feed, then displays the top three move suggestions for both White and Black by drawing colored arrows on the board.

Additionally, a test script is provided for calibrating the chessboard by selecting its four corners.

> **Note:** I used the Iriun Webcam App on both my phone and pc to use my phone as a camera, so the setup works with both built-in webcams and external devices.

## Features

- **Real-Time Chess Piece Detection**  
  Uses a YOLO model to detect chess pieces from a live webcam feed.
  
- **FEN Generation & Stability Check**  
  Converts piece detections into a FEN string with a stability filter to reduce jitter.
  
- **Stockfish Integration**  
  Queries Stockfish for the top three move suggestions for both White (blue arrows) and Black (red arrows).
  
- **Visual Feedback**  
  Draws a chess grid and arrows representing the best moves on a warped, top-down view of the chessboard.
  
- **Chessboard Corner Selection**  
  A separate script (`edgeSelect.py`) lets you click and save the four corners of your chessboard for proper perspective warping.

## Installation

### Prerequisites

- Python 3.7 or higher
- A working external camera (e.g., via the Iriun Webcam App to use your phone as a camera) [Iriun website](https://iriun.com/)
- Make sure your smart phone and pc are connected to the same wifi network when running Iriun on your pc and phone
- Stockfish chess engine (download from [Stockfish website](https://stockfishchess.org/download/))
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
- **1. Select Chessboard Corners**
Run the following command in your terminal:

```bash
python edgeSelect.py
```
A window will open showing the webcam feed.
Click four points corresponding to the corners of the chessboard (top-left, top-right, bottom-left, bottom-right) in order.
Press `s` to save the corners.
The saved file (chessboard_corners.p) will be used for warping the board.
- **2. Run The Application**
After saving the corners,change the StockFish path
```python
stockfish = Stockfish(path="C:/Users/Presision/Downloads/stockfish-windows-x86-64/stockfish/stockfish-windows-x86-64.exe")
```
on line 27 in `main.py` to your own StockFish path. Then run:
```bash
python main.py
```
The application will start capturing the webcam feed.
The chessboard image is warped into a top-down view, and a grid is drawn.
YOLO detects chess pieces on the board and converts the detections into a FEN string.
Stockfish analyzes the board and the top three moves for both White (displayed as blue arrows) and Black (displayed as red arrows) are drawn on the warped image.
The original webcam feed and the warped chessboard are displayed in separate windows.
Press `q` to exit the application.

