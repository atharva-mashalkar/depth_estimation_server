import cv2
import torch
import time
import numpy as np

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def dep_est_load_model(model_type):
    """
    Load a MiDas model for depth estimation
    # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Large"
    # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "DPT_Hybrid"
    # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    model_type = "MiDaS_small"
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    # Move model to GPU if available
    midas.to(device)
    midas.eval()
    return midas


def dep_est_model_transformation(model_type):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return transform


def find_depth(img, midas, transforms, coordinates):
    # Starting the timer
    start = time.time()

    # Getting coordinates of the required part of the image
    [x_start, y_start, x_end, y_end, x, y] = coordinates

    # Apply input transforms
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transforms(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # Finding finter tip coordinates
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    x_start, x_end = max(0, x_start), min(img.shape[1]-1, x_end)
    y_start, y_end = max(0, y_start), min(img.shape[0]-1, y_end)
    image_crop = depth_map[y_start:y_end, x_start:x_end]
    if x and y:
        print(x, y)
        print(depth_map[y, x, :])
        # print(depth_map[:, :, 2][y, x], img[:, :, 1][y, x], img[:, :, 0][y, x])

    # getMaxHeat(image_crop)
    # Calculating FPS
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    # Displaying Video
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    if x_end-x_start > 0 and y_end-y_start > 0:
        cv2.imshow('Croped_image', image_crop)
    cv2.imshow('Depth Map', depth_map)


def getMaxHeat(img):
    # extract red channel
    red_channel = img[:, :, 2]
    if len(red_channel) > 0:
        i, j = np.unravel_index(red_channel.argmax(), red_channel.shape)
        print(i, j)
        print(red_channel[i][j])
