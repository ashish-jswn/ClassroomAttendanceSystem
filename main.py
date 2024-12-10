# File path: main.py

import cv2
import numpy as np
import pandas as pd
import pickle
import os
import time
from insightface.app import FaceAnalysis  # RetinaFace
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load pre-trained RetinaFace model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # GPUExecutionProvider for GPU
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load face encodings from the pickle file
with open('face_encodings.pickle', 'rb') as f:
    student_encodings = pickle.load(f)


def detect_faces_retina(image):
    """
    Detect faces using RetinaFace.
    :param image: Input image as a NumPy array (BGR format).
    :return: Detected face bounding boxes and aligned cropped faces.
    """
    faces = face_app.get(image)  # Perform face detection
    face_bounding_boxes = []
    aligned_faces = []

    for face in faces:
        bbox = face.bbox.astype(int)  # Bounding box [x1, y1, x2, y2]
        face_bounding_boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
        aligned_faces.append(face.normed_embedding)  # Aligned cropped face

    return face_bounding_boxes, aligned_faces


def recognize_faces(image):
    """
    Recognize faces in a given image using RetinaFace and ArcFace embeddings.
    :param image: Input image as a NumPy array.
    :return: Annotated image and recognition results.
    """
    face_locations, aligned_faces = detect_faces_retina(image)
    results = []
    confidence_threshold = 0.5

    for (x1, y1, x2, y2), aligned_face in zip(face_locations, aligned_faces):
        best_match_distance = -1  # Cosine similarity: closer to 1 is better
        matched_roll_number = "Unknown"

        # Compare with all stored student encodings
        for roll_number, student_data in student_encodings.items():
            stored_embeddings = student_data['encodings']
            similarities = cosine_similarity([aligned_face], stored_embeddings)
            max_similarity = np.max(similarities)

            if max_similarity > best_match_distance and max_similarity > confidence_threshold:
                best_match_distance = max_similarity
                matched_roll_number = roll_number

        # Annotate results
        if matched_roll_number != "Unknown":
            results.append((x1, y1, x2, y2, matched_roll_number))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{matched_roll_number}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            results.append((x1, y1, x2, y2, "Unknown"))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image, results


def generate_attendance_csv(results, student_encodings, output_csv_path):

    current_date = datetime.now().date()

    attendance = {roll_number: 'A' for roll_number in student_encodings.keys()}
    for _, _, _, _, roll_number in results:
        if roll_number != "Unknown":
            attendance[roll_number] = 'P'
    attendance_df = pd.DataFrame(list(attendance.items()), columns=['Roll Number', str(current_date)])
    attendance_df.to_csv(output_csv_path, index=False)
    print(f"Attendance has been saved to {output_csv_path}")


def capture_stable_image_from_camera(timeout=7, stability_threshold=3):
    """
    Capture an image from the webcam after detecting stable faces.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # # Add a delay before capturing any frames
    # print("Switching on the camera. Waiting for 3 seconds...")
    # time.sleep(3)  # 3-second delay

    print(f"Switching on the camera. Detecting faces for up to {timeout} seconds...")
    start_time = time.time()
    stable_frame_count = 0
    stable_frame = None

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Detect faces using RetinaFace
        faces = detect_faces_retina(frame)

        # Draw rectangles for visualization
        for (x1, y1, x2, y2) in faces[0]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show the current frame
        cv2.imshow("Camera Feed - Detecting Faces", frame)

        # Check stability
        if len(faces[0]) > 0:
            stable_frame_count += 1
            stable_frame = frame
        else:
            stable_frame_count = 0

        # If stable frames meet the threshold, capture the image
        if stable_frame_count >= stability_threshold:
            print("Stable frame detected. Capturing image.")
            break

        # Exit gracefully on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if stable_frame is None:
        print("No stable frame detected. Returning the last available frame.")
        return frame

    return stable_frame


def process_option(option):
    if option == 1:
        input_image_path = input("Enter the path to the input image: ").strip()
        if not os.path.isfile(input_image_path):
            print(f"Error: File '{input_image_path}' does not exist.")
            return
        input_image = cv2.imread(input_image_path)
        processed_image, results = recognize_faces(input_image)
        generate_attendance_csv(results, student_encodings, 'attendance.csv')
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif option == 2:
        captured_image = capture_stable_image_from_camera(timeout=7, stability_threshold=3)
        if captured_image is None:
            return
        processed_image, results = recognize_faces(captured_image)
        generate_attendance_csv(results, student_encodings, 'attendance.csv')
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid option. Please choose either 1 or 2.")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Process an input image")
    print("2. Capture an image using the laptop's camera")
    try:
        user_option = int(input("Enter your choice (1 or 2): "))
        process_option(user_option)
    except ValueError:
        print("Invalid input. Please enter 1 or 2.")



