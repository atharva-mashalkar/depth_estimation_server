import cv2
from depth_estimation import dep_est_load_model, dep_est_model_transformation, find_depth
from parse_arg import parse_args
from hand_rec import load_hand_rec_model, find_index_fing_tip

# Parse arguments
args = parse_args()
cap_device = args.device
cap_width = args.width
cap_height = args.height
use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

# Load depth estimation model and necessary transformations to resize and normalize the image
# MODEL_TYPE = "MiDaS_small"  # "DPT_Hybrid" "DPT_Large"
# dep_est_model = dep_est_load_model(MODEL_TYPE)
# dep_est_transforms = dep_est_model_transformation(MODEL_TYPE)

# Load hand_recognition model
# hands = load_hand_rec_model(use_static_image_mode,
#                             min_detection_confidence, min_tracking_confidence)

# Camera preparation ###############################################################
# cap = cv2.VideoCapture(cap_device)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         break

#     final_img, coordinates = find_index_fing_tip(img, hands)
#     find_depth(final_img, dep_est_model, dep_est_transforms, coordinates)

#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
