# -*- coding: utf-8 -*-
import logging
from crnn_ocr.predict import crnnPredictor



class OCR(object):
    def __init__(self):
        self.config = {
            'char_dict_path': '/home/zhangjiacheng/ai_image/table/form_recognition/src/crnn_ocr/dict/char_std_5990.txt',
            'img_path': '/home/zhangjiacheng/data/table/chinese_ocr/test_imgs/hangzhou_imgs/cell_9.jpg',
            'model_path': '/home/zhangjiacheng/data/table/chinese_ocr/models/weights_densenet.h5',
            'img_h': 32,
            'img_w': 280,
        }
        self.crnn_predictor = crnnPredictor(self.config)

    def img_enhancement(self, image):
        # TODO: image enhancement
        enhancemened_image = image
        return enhancemened_image

    def text_recognition(self, image):
        output = self.crnn_predictor.run(image)
        return output
