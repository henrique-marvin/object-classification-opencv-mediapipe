import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import numpy as np
import random

# setting the model path
model_path = "efficientdet_lite0.tflite"

# setting the path to the image file
image_path = "path/to/image"  # example: C:/Users/Me/Images/Robot.jpg

# setting the window name, for when it's opened
window_name = "my image"  # could be anything

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # generating different border colors
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        TEXT_COLOR = (r, g, b)  # border color

        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv.rectangle(image, start_point, end_point, TEXT_COLOR, 6)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv.putText(
            image,
            result_text,
            text_location,
            cv.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=-1,
)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file(image_path)

detection_result = detector.detect(image)

image_copy = np.copy(image.numpy_view())

detection_image = visualize(image_copy, detection_result)

rgb_detection_image = cv.cvtColor(detection_image, cv.COLOR_BGR2RGB)

cv.imshow(window_name, rgb_detection_image)

# setting the fps -> in this case is 0, because it's a static image
button = cv.waitKey(0)

# using the button to close the window
if button == 27:  # 27 is ESC button, according to ASCII table
    cv.destroyAllWindows()
