import cv2
import pickle

# Webcam settings
cam_id = 1  # Change if needed
width, height = 1280, 720

# Open webcam
cap = cv2.VideoCapture(cam_id)
cap.set(3, width)
cap.set(4, height)

# List to store clicked points
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")

# Display webcam feed and allow user to click four corners
while True:
    success, img = cap.read()
    if not success:
        break

    for point in points:
        cv2.circle(img, point, 5, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Select Chessboard Corners", img)
    cv2.setMouseCallback("Select Chessboard Corners", mouse_callback)

    # Press 's' to save points
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if len(points) == 4:
            with open("chessboard_corners.p", "wb") as f:
                pickle.dump(points, f)
            print("Saved chessboard corners!")
            break
        else:
            print("Please select exactly 4 points.")

cap.release()
cv2.destroyAllWindows()
