import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def extract_eyes_from_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    eyes = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                        for landmark in face_landmarks.landmark[474:478]]
            right_eye = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                         for landmark in face_landmarks.landmark[469:473]]

            for eye in [left_eye, right_eye]:
                x_coords = [point[0] for point in eye]
                y_coords = [point[1] for point in eye]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                cropped_eye = image[y_min:y_max, x_min:x_max]
                if cropped_eye.size > 0:
                    eyes.append(cropped_eye)
    return eyes


