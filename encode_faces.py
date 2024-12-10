
# import os
# import pandas as pd
# import pickle
# import numpy as np
# from multiprocessing import Pool, cpu_count
# from insightface.app import FaceAnalysis
# import cv2

# # Load pre-trained RetinaFace model
# face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# # Load student data
# students_df = pd.read_csv('student.csv')

# # Initialize lists to store paths of images with no detected faces and augmented images directory
# images_no_faces_detected = []
# augmented_images_dir = "augmented_images"
# os.makedirs(augmented_images_dir, exist_ok=True)


# def augment_face(image, face_bbox):
#     """
#     Generate augmented versions of a face region in an image.
#     """
#     x1, y1, x2, y2 = map(int, face_bbox)
#     face = image[y1:y2, x1:x2]
#     height, width = face.shape[:2]

#     augmented_faces = []

#     # Cover half part of the face from left
#     left_covered = face.copy()
#     left_covered[:, :width // 2] = 0
#     augmented_faces.append(left_covered)

#     # Cover half part of the face from right
#     right_covered = face.copy()
#     right_covered[:, width // 2:] = 0
#     augmented_faces.append(right_covered)

#     # Cover half part of the face from down
#     down_covered = face.copy()
#     down_covered[height // 2:, :] = 0
#     augmented_faces.append(down_covered)

#     return augmented_faces


# def process_student_images(roll_number, student_name, folder_path):
#     """
#     Process all images for a single student, generate face embeddings, and augmented embeddings.
#     """
#     encodings_list = []
#     if not os.path.isdir(folder_path):
#         print(f"Folder missing for roll number {roll_number}: {folder_path}")
#         return None

#     for image_name in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, image_name)
#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue

#             faces = face_app.get(image)
#             if not faces:
#                 images_no_faces_detected.append(image_path)

#             for face in faces:
#                 # Original face embedding
#                 encodings_list.append(face.normed_embedding)

#                 # Generate augmented images
#                 augmented_faces = augment_face(image, face.bbox)

#                 for idx, aug_face in enumerate(augmented_faces):
#                     # Extract embedding for each augmented face
#                     aug_embedding = face_app.get(aug_face)
#                     if aug_embedding:
#                         encodings_list.append(aug_embedding[0].normed_embedding)

#                     # Save augmented images for inspection
#                     aug_image_name = f"{roll_number}_{os.path.splitext(image_name)[0]}_aug{idx+1}.jpg"
#                     cv2.imwrite(os.path.join(augmented_images_dir, aug_image_name), aug_face)

#         except Exception as e:
#             print(f"Error processing image {image_path}: {e}")

#     if encodings_list:
#         return {
#             'roll_number': roll_number,
#             'name': student_name,
#             'encodings': encodings_list
#         }
#     else:
#         print(f"No valid encodings found for roll number {roll_number}")
#         return None


# def process_student_wrapper(args):
#     return process_student_images(*args)


# if __name__ == "__main__":
#     student_encodings = {}

#     # Prepare arguments for multiprocessing
#     tasks = [
#         (row['roll_number'], row['name'], f"student_images/{row['roll_number']}")
#         for _, row in students_df.iterrows()
#     ]

#     print("Processing student images in parallel...")
#     with Pool(cpu_count()) as pool:
#         results = pool.map(process_student_wrapper, tasks)

#     for result in results:
#         if result is not None:
#             student_encodings[result['roll_number']] = {
#                 'name': result['name'],
#                 'encodings': result['encodings']
#             }

#     # Save encodings to a file
#     with open('face_encodings.pickle', 'wb') as f:
#         pickle.dump(student_encodings, f)

#     print("Face encodings have been saved to 'face_encodings.pickle'.")

#     # Print paths of images where no faces were detected
#     if images_no_faces_detected:
#         print("\nImages with no detected faces:")
#         for path in images_no_faces_detected:
#             print(path)
#     else:
#         print("\nAll images had faces detected.")


# import os
# import pandas as pd
# import pickle
# import numpy as np
# from multiprocessing import Pool, cpu_count
# from insightface.app import FaceAnalysis
# import cv2

# # Load pre-trained RetinaFace model
# face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# # Load student data
# students_df = pd.read_csv('student.csv')

# # Initialize lists to store paths of images with no detected faces and augmented images directory
# images_no_faces_detected = []
# augmented_images_dir = "augmented_images"
# os.makedirs(augmented_images_dir, exist_ok=True)


# def augment_face(image, face_bbox):
#     """
#     Generate augmented versions of a face region in an image.
#     """
#     x1, y1, x2, y2 = map(int, face_bbox)
#     face = image[y1:y2, x1:x2]
#     height, width = face.shape[:2]

#     augmented_faces = []

#     # Cover half part of the face from left
#     left_covered = face.copy()
#     left_covered[:, :width // 2] = 0
#     augmented_faces.append(left_covered)

#     # Cover half part of the face from right
#     right_covered = face.copy()
#     right_covered[:, width // 2:] = 0
#     augmented_faces.append(right_covered)

#     # Cover half part of the face from down
#     down_covered = face.copy()
#     down_covered[height // 2:, :] = 0
#     augmented_faces.append(down_covered)

#     return augmented_faces


# def process_student_images(roll_number, student_name, folder_path):
#     """
#     Process all images for a single student, generate face embeddings, and augmented embeddings.
#     """
#     encodings_list = []
#     if not os.path.isdir(folder_path):
#         print(f"Folder missing for roll number {roll_number}: {folder_path}")
#         return None

#     for image_name in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, image_name)
#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue

#             faces = face_app.get(image)
#             if not faces:
#                 print(f"No face detected in image: {image_path}")
#                 images_no_faces_detected.append(image_path)
#                 continue
#             else:
#                 print(f"Face detected in image: {image_path}")

#             for face in faces:
#                 # Original face embedding
#                 encodings_list.append(face.normed_embedding)
#                 print(f"Encoding generated for face in image: {image_path}")

#                 # Generate augmented images
#                 augmented_faces = augment_face(image, face.bbox)
#                 for idx, aug_face in enumerate(augmented_faces):
#                     # Extract embedding for each augmented face
#                     aug_embedding = face_app.get(aug_face)
#                     if aug_embedding:
#                         encodings_list.append(aug_embedding[0].normed_embedding)
#                         print(f"Encoding generated for augmented face {idx+1} in image: {image_path}")

#                     # Save augmented images for inspection
#                     aug_image_name = f"{roll_number}_{os.path.splitext(image_name)[0]}_aug{idx+1}.jpg"
#                     aug_image_path = os.path.join(augmented_images_dir, aug_image_name)
#                     cv2.imwrite(aug_image_path, aug_face)
#                     print(f"Augmented image saved: {aug_image_path}")

#         except Exception as e:
#             print(f"Error processing image {image_path}: {e}")

#     if encodings_list:
#         return {
#             'roll_number': roll_number,
#             'name': student_name,
#             'encodings': encodings_list
#         }
#     else:
#         print(f"No valid encodings found for roll number {roll_number}")
#         return None


# def process_student_wrapper(args):
#     return process_student_images(*args)


# if __name__ == "__main__":
#     student_encodings = {}

#     # Prepare arguments for multiprocessing
#     tasks = [
#         (row['roll_number'], row['name'], f"student_images/{row['roll_number']}")
#         for _, row in students_df.iterrows()
#     ]

#     print("Processing student images in parallel...")
#     with Pool(cpu_count()) as pool:
#         results = pool.map(process_student_wrapper, tasks)

#     for result in results:
#         if result is not None:
#             student_encodings[result['roll_number']] = {
#                 'name': result['name'],
#                 'encodings': result['encodings']
#             }

#     # Save encodings to a file
#     with open('face_encodings.pickle', 'wb') as f:
#         pickle.dump(student_encodings, f)

#     print("Face encodings have been saved to 'face_encodings.pickle'.")

#     # Print paths of images where no faces were detected
#     if images_no_faces_detected:
#         print("\nImages with no detected faces:")
#         for path in images_no_faces_detected:
#             print(path)
#     else:
#         print("\nAll images had faces detected.")


import os
import pandas as pd
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from insightface.app import FaceAnalysis
import cv2

# Load pre-trained RetinaFace model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load student data
students_df = pd.read_csv('student.csv')

# Initialize augmented images directory
augmented_images_dir = "augmented_images"
os.makedirs(augmented_images_dir, exist_ok=True)


def augment_face(image, face_bbox):
    """
    Generate augmented versions of a face region in an image.
    """
    x1, y1, x2, y2 = map(int, face_bbox)
    face = image[y1:y2, x1:x2]
    height, width = face.shape[:2]

    augmented_faces = []

    # Cover half part of the face from left
    left_covered = face.copy()
    left_covered[:, :width // 2] = 0
    augmented_faces.append(left_covered)

    # Cover half part of the face from right
    right_covered = face.copy()
    right_covered[:, width // 2:] = 0
    augmented_faces.append(right_covered)

    # Cover half part of the face from down
    down_covered = face.copy()
    down_covered[height // 2:, :] = 0
    augmented_faces.append(down_covered)

    return augmented_faces


def process_student_images(args):
    """
    Process all images for a single student, generate face embeddings, and augmented embeddings.
    """
    roll_number, student_name, folder_path, no_faces_list = args
    encodings_list = []

    if not os.path.isdir(folder_path):
        print(f"Folder missing for roll number {roll_number}: {folder_path}")
        return None

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue

            faces = face_app.get(image)
            if not faces:
                print(f"No face detected in image: {image_path}")
                no_faces_list.append(image_path)
                continue
            else:
                print(f"Face detected in image: {image_path}")

            for face in faces:
                # Original face embedding
                encodings_list.append(face.normed_embedding)
                print(f"Encoding generated for face in image: {image_path}")

                # Generate augmented images
                augmented_faces = augment_face(image, face.bbox)
                for idx, aug_face in enumerate(augmented_faces):
                    # Extract embedding for each augmented face
                    aug_embedding = face_app.get(aug_face)
                    if aug_embedding:
                        encodings_list.append(aug_embedding[0].normed_embedding)
                        print(f"Encoding generated for augmented face {idx+1} in image: {image_path}")

                    # Save augmented images for inspection
                    aug_image_name = f"{roll_number}_{os.path.splitext(image_name)[0]}_aug{idx+1}.jpg"
                    aug_image_path = os.path.join(augmented_images_dir, aug_image_name)
                    cv2.imwrite(aug_image_path, aug_face)
                    print(f"Augmented image saved: {aug_image_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if encodings_list:
        return {
            'roll_number': roll_number,
            'name': student_name,
            'encodings': encodings_list
        }
    else:
        print(f"No valid encodings found for roll number {roll_number}")
        return None


if __name__ == "__main__":
    student_encodings = {}

    # Use Manager to create a shared list for images with no detected faces
    with Manager() as manager:
        images_no_faces_detected = manager.list()

        # Prepare arguments for multiprocessing
        tasks = [
            (row['roll_number'], row['name'], f"student_images/{row['roll_number']}", images_no_faces_detected)
            for _, row in students_df.iterrows()
        ]

        print("Processing student images in parallel...")
        with Pool(cpu_count()) as pool:
            results = pool.map(process_student_images, tasks)

        for result in results:
            if result is not None:
                student_encodings[result['roll_number']] = {
                    'name': result['name'],
                    'encodings': result['encodings']
                }

        # Save encodings to a file
        with open('face_encodings.pickle', 'wb') as f:
            pickle.dump(student_encodings, f)

        print("Face encodings have been saved to 'face_encodings.pickle'.")

        # Print paths of images where no faces were detected
        if images_no_faces_detected:
            print("\nImages with no detected faces:")
            for path in images_no_faces_detected:
                print(path)
        else:
            print("\nNo images were found without faces detected.")
