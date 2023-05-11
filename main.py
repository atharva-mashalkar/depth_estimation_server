import cv2
from depth_estimation import dep_est_load_model, dep_est_model_transformation, find_depth
from parse_arg import parse_args
from hand_rec import load_hand_rec_model, find_index_fing_tip
import base64
from imageio import imread
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# Parse arguments
args = parse_args()
cap_device = args.device
cap_width = args.width
cap_height = args.height
use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

# Load depth estimation model and necessary transformations to resize and normalize the image
MODEL_TYPE = "MiDaS_small"  # "DPT_Hybrid" "DPT_Large"
dep_est_model = dep_est_load_model(MODEL_TYPE)
dep_est_transforms = dep_est_model_transformation(MODEL_TYPE)

# Load hand_recognition model
hands = load_hand_rec_model(use_static_image_mode,
                            min_detection_confidence, min_tracking_confidence)

# Camera preparation ###############################################################
# cap = cv2.VideoCapture(cap_device)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


def runModels(msg):
    if msg:
        # Starting the timer
        start = time.time()

        # Reconstruct image as an numpy array
        img = imread(io.BytesIO(base64.b64decode(msg[23:])))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Getting Predictions
        final_img, coordinates = find_index_fing_tip(img, hands)
        output = find_depth(final_img, dep_est_model, dep_est_transforms, coordinates)
        # print(output)
        # return {
        #     'output': output
        # }

        bbCor, fingerTipCor, originaImg, depthMap,  maxHeat = output['bbCor'], output[
            'fingerTipCor'], output['originaImg'], output['depthMap'],  output['maxHeat']

        image_crop = depthMap[bbCor[1]:bbCor[3], bbCor[0]:bbCor[2]]

        # if fingerTipCor[0] and fingerTipCor[1]:
        #     print(fingerTipCor)
        #     print(depthMap[fingerTipCor[1], fingerTipCor[0]:])

        # Calculating FPS
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        # Imprinting FPS
        cv2.putText(originaImg, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Encoding images and sending them to client
        originaImg = 'data:image/jpeg;base64,' + \
            base64.b64encode(cv2.imencode('.jpeg', originaImg)[1]).decode()
        depthMap = 'data:image/jpeg;base64,' + \
            base64.b64encode(cv2.imencode('.jpeg', depthMap)[1]).decode()

        if bbCor[2]-bbCor[0] > 0 and bbCor[3]-bbCor[1] > 0:
            image_crop = 'data:image/jpeg;base64,' + \
                base64.b64encode(cv2.imencode('.jpeg', image_crop)[1]).decode()
        else:
            image_crop = ""

        # print(image_crop)
        return {'originaImg': originaImg, 'depthMap': depthMap, 'croppedImg': image_crop, 'maxHeat':maxHeat}
