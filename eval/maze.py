import json
import os
import random
import time
import re
import io
import numpy as np
from skimage.draw import line
from PIL import Image
import base64
import cv2
import copy


def get_maze_bounds(gray_array, white_threshold=250):
    mask = gray_array < white_threshold  # 非白区域
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("未检测到迷宫区域")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, x_max, y_min, y_max


def is_valid_step(gray_array, point1, point2, threshold=45):
    valid = True
    reason = "合法"

    height, width = gray_array.shape

    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])

    # 获取迷宫有效区域边界
    x_min, x_max, y_min, y_max = get_maze_bounds(gray_array)

    # 路径像素点判断
    rr, cc = line(y1, x1, y2, x2)

    rr = np.clip(rr, 0, height - 1)
    cc = np.clip(cc, 0, width - 1)

    # 越界
    if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
        return False, "起点终点不在图像范围内"

    # 起点终点在边界外（外部白边区域）
    for (x, y) in [(x1, y1), (x2, y2)]:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False, "起点终点在外围白边区域"
        if gray_array[y, x] < threshold:
            return False, "起点终点在墙上"

    for r, c in zip(rr, cc):
        if not (x_min <= c <= x_max and y_min <= r <= y_max):
            valid = False
            reason = "路径穿过外围白边区域"
            break
        if gray_array[r, c] < threshold:
            valid = False
            reason = "路径穿过墙"
            break

    return valid, reason


def extract_maze_answer(pred_str):
    pred_str = re.findall(r"(?:<point>\d+\s+\d+</point>)+", pred_str)
    if len(pred_str) > 0:
        pred_str = pred_str[-1]
    else:
        pred_str = ""
    line_pattern = re.compile(r'<point>(.*?)</point>', re.DOTALL)
    line_contents = line_pattern.findall(pred_str)
    try:
        line_contents = [(int(_.split(' ')[0]), int(_.split(' ')[1])) for _ in line_contents]
        line_contents = [(int(_[0] / 1000 * 512), int(_[1] / 1000 * 512)) for _ in line_contents]
    except:
        return []
    return line_contents


def wall_judge(response, image_base64, answer, maze_size="5"):

    results_gt = list(map(int, re.findall(r"\d+", answer)))
    results_gt_new = []
    for i in range(0, len(results_gt), 2):
        results_gt_new.append([results_gt[i], results_gt[i + 1]])

    point_list_new = extract_maze_answer(response)

    # load wall info
    if type(image_base64) is str:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = image_base64

    gray_array = np.array(image.convert('RGB')).max(-1)
    drawed_gray_array = copy.deepcopy(gray_array)

    if len(point_list_new) <= 1:
        return 0, drawed_gray_array

    # draw lines on gray_array
    for i in range(1, len(point_list_new)):
        drawed_gray_array = cv2.line(drawed_gray_array, point_list_new[i - 1], point_list_new[i], (0, 0, 0), 2)

    # judge if crossing wall
    last_point = point_list_new[0]
    first_point = point_list_new[0]
    for i in range(1, len(point_list_new)):
        judge, reason = is_valid_step(gray_array, point_list_new[i - 1], point_list_new[i])
        if judge == False:
            last_point = point_list_new[i - 1]
            # The reward for crossing a wall is 0.
            return 0, drawed_gray_array

        if i == len(point_list_new) - 1:
            last_point = point_list_new[i]

    # find the closest steps in gt
    index = 0

    if maze_size == "5":
        min_delta = 100
    else:
        raise NotImplementedError

    min_delta = min_delta / 1500 * max(gray_array.shape)

    dist_from_start = np.sqrt((first_point[0] - results_gt_new[0][0]) ** 2 +
                              (first_point[1] - results_gt_new[0][1])**2 + 0.01)
    if dist_from_start >= min_delta:
        return 0, drawed_gray_array

    for i, (px, py) in enumerate(results_gt_new):
        dist = np.sqrt((px - last_point[0]) ** 2 + (py - last_point[1])**2 + 0.01)
        if dist < min_delta:
            min_delta = dist
            index = i

    return index / (len(results_gt_new) - 1), drawed_gray_array
