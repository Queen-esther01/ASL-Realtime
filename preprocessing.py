import os
import glob
import cv2 as cv
import pandas as pd
import mediapipe as mp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LandmarkExtractor:
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (0, 17)  # Palm base
    ]

    def __init__(self, model_path, dataset_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE
        )

    def extract_landmarks(self):
        landmarks = []
        images = []
        rows = []

        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            for file in sorted(glob.glob(self.dataset_path)):
                # if directories do not exist, create them
                if not os.path.exists('data/clean_images'):
                    os.makedirs('data/clean_images')
                if not os.path.exists('data/bad_images'):
                    os.makedirs('data/bad_images')
                stemFileName = (Path(os.path.basename(file)).stem)  # get folder name A, B, C...
                if not os.path.exists(f'data/clean_images/{stemFileName}'):
                    os.makedirs(f'data/clean_images/{stemFileName}')
                if not os.path.exists(f'data/bad_images/{stemFileName}'):
                    os.makedirs(f'data/bad_images/{stemFileName}')

                # LOOP THROUGH INNER FOLDER
                for item in sorted(glob.glob(file + '/*')):
                    filename = item.split('/')[-1]
                    image = cv.imread(item)
                    mp_image = mp.Image.create_from_file(item)
                    hand_landmarker_result = landmarker.detect(mp_image)
                    landmark_per_hand = []
                    if hand_landmarker_result.hand_landmarks:  # Focus on only images with landmarks
                        image_landmarks = []
                        images.append(image)
                        for hand_landmarks in hand_landmarker_result.hand_landmarks:
                            x = [float(landmark.x) for landmark in hand_landmarks]
                            y = [float(landmark.y) for landmark in hand_landmarks]
                            for landmark in hand_landmarks:
                                landmark_per_hand.append([float(landmark.x), float(landmark.y), float(landmark.z)])
                            image_landmarks.append(landmark_per_hand)
                        # SAVE GOOD IMAGES WITH LANDMARKS
                        image_with_landmarks = self.draw_hand_landmarks(image, image_landmarks)
                        self.save_image(f'data/clean_images/{stemFileName}/{filename}', image_with_landmarks)
                        rows.append({
                            'landmark': landmark_per_hand,
                            'x': x,
                            'y': y,
                            'label': stemFileName,
                        })
                        landmarks.append(landmark_per_hand)
                    else:
                        # SAVE BAD IMAGES W/O LANDMARKS
                        self.save_image(f'data/bad_images/{stemFileName}/{filename}', image)
        # SAVE DATA TO CSV
        if not os.path.exists('data/clean_dataset'):
            os.makedirs('data/clean_dataset')
        df = pd.DataFrame(rows)
        df.to_csv(f'data/clean_dataset/data.csv')
        return landmarks

    def draw_hand_landmarks(self, image, hand_landmarks):
        '''Draws landmarks on the input image and returns it.'''
        # image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        h, w = image.shape[:2]
        points = []
        for hand_landmarks in hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm[0] * w)
                y = int(lm[1] * h)
                points.append((x, y))

        # Draw connections
        for start, end in self.HAND_CONNECTIONS:
            cv.line(image, points[start], points[end], (0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        # Draw Landmarks
        for x, y in points:
            cv.circle(image, (x, y), 4, (255, 0, 0), -1)

        return image

    def save_image(self, path, image):
        cv.imwrite(path, image)

    def visualize_landmarks(self, image):
        landmarks = [[[0.744321346282959, 0.7614148259162903, -9.449505569136818e-07],
                      [0.8156479001045227, 0.5896764993667603, -0.05400242283940315],
                      [0.7811483144760132, 0.41055363416671753, -0.07408493757247925],
                      [0.6172792911529541, 0.3630075454711914, -0.09375110268592834],
                      [0.4892432391643524, 0.4260185956954956, -0.09972097724676132],
                      [0.6303133964538574, 0.3241669833660126, -0.03299500048160553],
                      [0.518669843673706, 0.2205173373222351, -0.09953555464744568],
                      [0.562858521938324, 0.3445088863372803, -0.12997059524059296],
                      [0.6186015009880066, 0.4195975065231323, -0.14179393649101257],
                      [0.5304473042488098, 0.40138471126556396, -0.04285213351249695],
                      [0.41314229369163513, 0.2965489327907562, -0.12662272155284882],
                      [0.4937394857406616, 0.4337039887905121, -0.1505899429321289],
                      [0.564757227897644, 0.49199360609054565, -0.1505458652973175],
                      [0.44845250248908997, 0.48737436532974243, -0.06527204811573029],
                      [0.3360994756221771, 0.40337640047073364, -0.15217548608779907],
                      [0.4348604083061218, 0.5006554126739502, -0.1655098795890808],
                      [0.5212993025779724, 0.5531733632087708, -0.1546841859817505],
                      [0.38490188121795654, 0.5772625803947449, -0.09134108573198318],
                      [0.26065412163734436, 0.4996018409729004, -0.1407497078180313],
                      [0.18603703379631042, 0.4543452858924866, -0.15747776627540588],
                      [0.10997521877288818, 0.399292528629303, -0.16567575931549072]]]
        color_image = cv.imread(image, cv.IMREAD_COLOR_RGB)
        grayscale = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
        image_with_landmarks = self.draw_hand_landmarks(grayscale, landmarks)
        plt.imshow(image_with_landmarks)
        plt.show()


if __name__ == '__main__':
    extractor = LandmarkExtractor('mediapipe/hand_landmarker.task', 'data/CW2_dataset_final/*')
    landmarks = extractor.extract_landmarks()
