import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = './Model/gesture_recognizer.task'
video = cv2.VideoCapture(0)


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # cv2.imshow('Show2', output_image.numpy_view())
    # cv2.imshow почему-то не работает в callback, пришлось выкидывать результат в глобальную область
    global category

    category = ''
    if result.gestures:
        gest = result.gestures[0][0]
        category = gest.category_name


category = ''
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 400)
fontScale = 1
color = (255, 0, 0)
thickness = 2
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    while cv2.waitKey(1) < 0:
        ret, frame = video.read()
        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)
        if category and category != 'None':
            frame = cv2.putText(frame, category, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Show', frame)


video.release()
cv2.destroyAllWindows()
