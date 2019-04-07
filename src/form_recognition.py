# -*- coding: utf-8 -*-

import cv2
import logging


class formRecognition(object):
    def __init__(self, image):
        self.image = image
        self.binary_img = None
        self.verticalLines_img = None
        self.horizontalLines_img = None
        self.form_list = []
        self.retval = 0
        self.labels = []
        self.stats = []
        self.centroids = []
        self.cell_img_list = []
        self.cells = []

    def color2binary(self):
        #转为灰度图
        #红色印章R通道值较大，白色纸张三个通道值都很大，因此将R通道作为灰度图
        gray_img = self.image[:, :, 2]

        #二值化
        ret, self.binary_img = cv2.threshold(~gray_img, 50, 255, cv2.THRESH_BINARY)
        return self.binary_img

    def detect_vertical_lines(self):
        # 开运算提取垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200))
        self.verticalLines_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, vertical_kernel)
        ret, self.verticalLines_img = cv2.threshold(self.verticalLines_img, 0, 255, cv2.THRESH_BINARY)
        return self.verticalLines_img

    def detect_horizontal_lines(self):
        # 开运算提取水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
        self.horizontalLines_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, horizontal_kernel)
        ret, self.horizontalLines_img = cv2.threshold(self.horizontalLines_img, 0, 255, cv2.THRESH_BINARY)
        return self.horizontalLines_img

    def mark_subtable(self):
        #标记出页面上所有表格
        form_lines_mask = self.horizontalLines_img + self.verticalLines_img           #合并出完整个表格
        contours, hier = cv2.findContours(form_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #提取所有外层轮廓
        for c in contours:
            #生成一个表格列表
            blank = form_lines_mask - form_lines_mask
            x, y, w, h = cv2.boundingRect(c)
            form_lines_mask = cv2.rectangle(form_lines_mask, (x, y), (x + w, y + h), 255, 3)
            rect_mask = cv2.rectangle(blank, (x, y), (x + w, y + h), 255, -1)
            cells_mask = rect_mask + (~form_lines_mask) - 255
            ret, form_cell_mask = cv2.threshold(cells_mask, 50, 255, cv2.THRESH_BINARY)
            self.form_list.append(form_cell_mask)
        return self.form_list

    def detect_cell(self, cell_img):
        self.retval, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(cell_img)
        for i in range(1, self.retval):
            x, y, w, h = self.stats[i][0:4]
            cell_img = self.image[y:y + h, x:x + w]
            self.cell_img_list.append(cell_img)
        return self.retval, self.labels, self.stats, self.centroids, self.cell_img_list

    def run(self):
        if self.image is not None:
            self.color2binary()

        if self.binary_img is not None:
            self.detect_vertical_lines()
            self.detect_horizontal_lines()
            self.mark_subtable()

        if self.form_list != []:
            for index, form in enumerate(self.form_list):
                self.detect_cell(form)
                self.cells.append(self.cell_img_list)
            return self.cells









