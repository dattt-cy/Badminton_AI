import cv2
import argparse
import sys
import numpy as np

corners = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            print(f"Clicked point {len(corners)}: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read video")
        sys.exit(1)

    window_name = "CLICK 4 COURT CORNERS (BL -> BR -> FR -> FL). PRESS 'q' WHEN DONE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_frame = frame.copy()
        
        # Draw circles and lines
        for i, pt in enumerate(corners):
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if i > 0:
                cv2.line(display_frame, corners[i-1], corners[i], (0, 255, 0), 2)
        if len(corners) == 4:
            cv2.line(display_frame, corners[3], corners[0], (0, 255, 0), 2)
            
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('q') or len(corners) == 4:
            # wait a bit for user to see the closed box
            if len(corners) == 4:
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(1000)
            break

    cv2.destroyAllWindows()

    if len(corners) == 4:
        with open(args.output, 'w') as f:
            f.write(','.join([f"{x},{y}" for x, y in corners]))
        print(f"Saved corners to {args.output}")
    else:
        print("Did not get 4 corners.")
        sys.exit(1)

if __name__ == "__main__":
    main()

