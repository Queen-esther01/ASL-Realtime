import os
import ast
import time
import numpy as np
import cv2
from sklearn import svm, preprocessing
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from cv_pipeline import pipeline, parameters
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

class HandPose:
    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.MARGIN = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 2
        self.HANDEDNESS_TEXT_COLOR = (0, 0, 255) # vibrant green
        # Create Global Variable
        self.latest_result = None
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', "H", "I", "J"]

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if len(detection_result.hand_landmarks) == 0:
            return rgb_image
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            self.mp_drawing.draw_landmarks(
              annotated_image,
              hand_landmarks,
              self.mp_hands.HAND_CONNECTIONS,
              self.mp_drawing_styles.get_default_hand_landmarks_style(),
              self.mp_drawing_styles.get_default_hand_connections_style()
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            # Run Model
            x_y_coordinates = np.concatenate((x_coordinates, y_coordinates), axis=0).reshape(1, -1)
            # x_y_coordinates = [x_coordinates, y_coordinates]
            prediction = self.test_best_model(x_y_coordinates)
            # print(f"shape: {x_y_coordinates.shape}")

            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
            # {handedness[0].category_name} -> Right or Left
            cv2.putText(annotated_image, f", Letter: {prediction[0]}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

            return annotated_image

    # Update Variable
    def save_result(self, result, output_image: mp.Image, timestamp_ms: int):
        global latest_result
        self.latest_result = result

    def find_landmarks(self):
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='mediapipe/hand_landmarker.task'),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_presence_confidence=0.8,
            result_callback=self.save_result)
        with self.HandLandmarker.create_from_options(options) as landmarker:
            video_capture = cv2.VideoCapture(0)
            window_name = 'Hand Landmarker Demo'
            cv2.namedWindow(window_name)
            while True:
                has_frame, frame = video_capture.read()
                if not has_frame:
                    break

                # Convert frame to MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                timestamp_ms = int(time.time() * 1000)

                # Detect landmarks
                landmarker.detect_async(mp_image, timestamp_ms)

                if self.latest_result:
                    frame = self.draw_landmarks_on_image(frame, self.latest_result)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows(window_name)

    def flatten_data(self):
        all_data = pd.read_csv('data/clean_dataset/data.csv')
        x = np.array(all_data['x'].apply(ast.literal_eval).tolist())
        y = np.array(all_data['y'].apply(ast.literal_eval).tolist())
        # print(f'x shape: {x.shape}, {x[0].shape}')
        # print(f'y shape: {y.shape}, {y[0].shape}')
        labels = all_data['label'].to_numpy()
        data = np.concatenate((x, y), axis=1)
        # print(f'data shape: {data.shape}, {data[0].shape}')
        # print(labels)
        return data, labels

    def initialize_grid_cv(self, model, parameters, X_train, y_train):
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search

    def svc_grid_cv(self):
        data, labels = self.flatten_data()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Initialize Model
        svc = svm.SVC()
        parameters = {
            'C': [0.1, 0.5, 1, 5, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

        # Train Model
        grid_search = self.initialize_grid_cv(svc, parameters, X_train, y_train)

        # Evaluate
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        print(f'SVM Accuracy: {accuracy}, best model: {best_model}, grid search: {grid_search.best_params_}')
        print(f'classification report: {report}\n')
        return grid_search.best_params_, accuracy, grid_search.best_score_

    def logistic_regression_grid_cv(self):
        data, labels = self.flatten_data()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Intialize Model
        model = LogisticRegression(max_iter=1000)
        parameters = {
            'C': [0.1, 0.5, 1, 5, 10],
        }

        # Encode Labels
        le = LabelEncoder()
        le.fit(self.classes)
        labels = le.transform(labels)
        print("labels:", labels[0], labels.shape)

        # Train Model
        grid_search = self.initialize_grid_cv(model, parameters, X_train, y_train)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        print(f'Logistic Regression Accuracy: {accuracy}, best model: {best_model}, grid search: {grid_search.best_params_}')
        print(f'classification report: {report}\n')
        return grid_search.best_params_, accuracy, grid_search.best_score_

    def train_best_model(self):
        data, labels = self.flatten_data()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        svc = svm.SVC(kernel='rbf', C=10)
        svc.fit(X_train, y_train)

        # if not os.path.exists('models'):
        #     os.makedirs('models')
        # joblib.dump(svc, 'models/best_model.pkl')

        # Evaluate
        predictions = svc.predict(X_test)
        score = svc.score(X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        print(f"fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}")
        print(f'svm score: {score}, accuracy: {accuracy}')
        print(f'classification report: {report}\n')

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d",
                    xticklabels=self.classes, yticklabels=self.classes, cmap="Blues")
        plt.title(f"Confusion Matrix: SVM, Accuracy: {accuracy * 100:.2f}%")

        # if not os.path.exists('reports'):
        #     os.makedirs('reports')
        # plt.savefig("reports/svm_cm.png")
        plt.close()
        return svc, score, accuracy

    def test_best_model(self, data):
        # data, labels = self.flatten_data()
        model = joblib.load('models/best_model.pkl')
        feature = data
        # feature = data[0:1]
        predictions = model.predict(feature)
        print(f'pred {predictions}')
        print(f'data: {feature}{feature.shape}')
        return predictions

if __name__ == '__main__':
    handPose = HandPose()
    # handPose.find_landmarks()
    # handPose.svc_grid_cv()
    # handPose.logistic_regression_grid_cv()
    handPose.train_best_model()
    # data, _ = handPose.flatten_data()
    # handPose.test_best_model(data)

