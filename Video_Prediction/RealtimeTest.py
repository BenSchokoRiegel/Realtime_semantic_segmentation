import datetime
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from Model import get_model, get_prediction

image_shape = [1024, 1024, 3]
rgb2classes = {
    (0, 0, 0): 0,  # Background (Schwarz)
    (0, 0, 255): 1,  # Human diver (Blau)
    (0, 255, 0): 2,  # Plant (Grün)
    (0, 255, 255): 3,  # Wreck or ruin (Sky)
    (255, 0, 0): 4,  # Robot (Rot)
    (255, 0, 255): 5,  # Reef or invertebrate (Pink)
    (255, 255, 0): 6,  # Fish or vertebrate (Gelb)
    (255, 255, 255): 7  # Sea-floor or rock (Weiß)
}
classColorMap = ListedColormap([(r / 255, g / 255, b / 255) for (r, g, b) in rgb2classes.keys()])


def start(video_path, model_path):
    model = get_model(model_path)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error reading video file")

    fig = plt.figure("Realtime Test")

    while True:
        ret, frame = video.read()

        if ret:
            frame = cv2.resize(frame, [1024, 1024])

            before_prediction = datetime.datetime.now()
            prediction = get_prediction(model, frame)
            dif = datetime.datetime.now() - before_prediction
            print("Prediction Time: " + str(int(dif.total_seconds() * 1000)) + " ms \n")

            plt.clf()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")
            plt.imshow(prediction.cpu(), alpha=0.5, interpolation="nearest", cmap=classColorMap)
            plt.pause(0.1)

        else:
            break

    video.release()
    cv2.destroyAllWindows()
