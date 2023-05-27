import mediapipe as mp
import cv2
import copy
from parse_arg import parse_args

args = parse_args()
cap_width = args.width
cap_height = args.height


def load_hand_rec_model(use_static_image_mode, min_detection_confidence, min_tracking_confidence):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return hands


def find_index_fing_tip(image, hands):
    # image = cv2.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)
    # Detection implementation #############################################################
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    #  ####################################################################
    x_start = y_start = x_end = y_end = 0
    w = h = int(cap_width*0.45)
    x = y = 0
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            # Bounding box calculation
            x, y = landmark_list[8][0], landmark_list[8][1]
            y_start = y - h//2
            y_end = y + h//2
            if handedness.classification[0].label[0:] == "Right":
                x_start, x_end = x+7, x+7+w
            else:
                x_start, x_end = x-w-7, x-7
            # debug_image = draw_bounding_rect(
            #     debug_image, [x_start, y_start, x_end, y_end])
    return debug_image, [x_start, y_start, x_end, y_end, x, y]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def draw_bounding_rect(image, brect):
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                  (0, 0, 0), 3)

    return image
