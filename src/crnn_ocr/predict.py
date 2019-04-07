# -*- coding: utf-8 -*-
from tensorflow import keras
from .dense_blstm import denseBlstm
import numpy as np
from PIL import Image
import cv2

class crnnPredictor():
    def __init__(self, config):
        self.char_dict_path = config['char_dict_path']
        self.model_path = config['model_path']
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.char_dict = None
        self.nclass = 0
        self.model = None
        self.create_dict()
        self.load_model()


    def create_dict(self):
        char = ''
        with open(self.char_dict_path, encoding='utf-8') as f:
            for ch in f.readlines():
                ch = ch.strip('\r\n')
                char = char + ch
        char = char[1:] + '卍'
        self.char_dict = {i: j for i, j in enumerate(char)}
        self.nclass = len(self.char_dict)
        return self.char_dict, self.nclass

    def load_model(self):
        dense_blstm = denseBlstm()
        base_input = keras.layers.Input(shape=(self.img_h, None, 1), name='the_input')
        base_output = dense_blstm.build_dense_blstm(base_input, self.nclass)
        self.model = keras.models.Model(inputs=base_input, outputs=base_output)
        self.model.summary()
        self.model.load_weights(self.model_path)
        return self.model

    def load_img(self, image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.convert('L')
        scale = img.size[1] * 1.0 / 32
        w = img.size[0] / scale
        w = int(w)
        print('w:', w)

        img = img.resize((w, self.img_h), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32) / 255.0 - 0.5
        x = img.reshape((self.img_h, w, 1))
        self.X = np.array([x])
        return self.X

    def predict(self):
        y_pred = self.model.predict(self.X)
        argmax = np.argmax(y_pred, axis=2)[0]
        y_pred = y_pred[:, :, :]
        output = keras.backend.get_value(keras.backend.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) *
                                                                                       y_pred.shape[1], )[0][0])[:, :]
        output = u''.join([self.char_dict[x] for x in output[0]])
        return output

    def run(self, image):
        self.load_img(image)
        output = self.predict()
        # print('predict result: ', output)
        return output


