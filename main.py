import cv2
import argparse

def load_cascades():
    """Load Haar cascades for face and eye detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face_cascade, eye_cascade

def detect_and_display_face(frame, gray_frame, face_cascade, eye_cascade):
    """Detect faces and eyes, display bounding boxes."""
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, width, height) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Detect eyes within the face region
        roi_gray = gray_frame[y:y + height, x:x + width]
        roi_color = frame[y:y + height, x:x + width]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_and_display_body(frame, hog_detector):
    """Detect full bodies using HOG-based human detector and display bounding boxes."""
    bodies, _ = hog_detector.detectMultiScale(frame, winStride=(8, 8))
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Body", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def start_camera(face_detection=True, body_detection=False):
    """Capture video from webcam and apply face, eye, or body detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    face_cascade, eye_cascade = load_cascades()
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_detection:
            detect_and_display_face(frame, gray_frame, face_cascade, eye_cascade)
        if body_detection:
            detect_and_display_body(frame, hog_detector)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Face, Eye, and Body Detection using OpenCV.")
    parser.add_argument('--face', action='store_true', help="Enable face detection")
    parser.add_argument('--body', action='store_true', help="Enable body detection")
    args = parser.parse_args()

    # By default, enable face detection unless specified otherwise
    face_detection = args.face or not args.body
    body_detection = args.body

    start_camera(face_detection=face_detection, body_detection=body_detection)
