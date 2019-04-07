import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def correct_img_bias(image):
    image = np.asarray(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(~gray_img, 100, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(binary_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img


def select_imgs_with_form(images):
    # TODO: 筛选PDF中含有表格的页面
    pass
