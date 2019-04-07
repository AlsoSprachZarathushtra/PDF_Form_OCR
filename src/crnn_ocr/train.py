# -*- coding: utf-8 -*-

from tensorflow import keras
from dense_blstm import denseBlstm
from PIL import Image
import os
import numpy as np


class crnnTrainer():
    def __init__(self, config):
        self.char_dict_path = config['char_dict_path']
        self.trainfile = config['trainfile']
        self.validationfile = config['validationfile']
        self.data_dir = config['data_dir']
        self.model_path = config['model_path']
        self.checkpoint_dir = config['checkpoint_dir']
        self.tensorboard_dir = config['tensorboard_dir']
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.maxlabellength = config['maxlabellength']
        self.batchsize = config['batchsize']
        self.epochs = config['epochs']
        self.steps_per_epoch = config['steps_per_epoch']
        self.validation_steps = config['validation_steps']
        self.char_dict = None
        self.nclass = 0
        self.model = None


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

    def random_uniform_num(self, n_samples, batchsize):
        index_list = [i for i in range(n_samples)]
        np.random.shuffle(index_list)
        index = 0
        r_n = []
        if index + batchsize > n_samples:
            r_n_1 = index_list[index:n_samples]
            np.random.shuffle(index_list)
            index = (index + batchsize) - n_samples
            r_n_2 = index_list[0:index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = index_list[index:index + batchsize]
        return r_n

    def readtrainfile(self, filename):
        res = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i in lines:
                res.append(i.strip('\r\n'))
        dic = {}
        for i in res:
            p = i.split(' ')
            dic[p[0]] = p[1:]
        return dic

    def gen(self, file):
        imagesize=(self.img_h, self.img_w)
        image_label = self.readtrainfile(file)
        _imagefile = [i for i, j in image_label.items()]
        x = np.zeros((self.batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
        labels = np.ones([self.batchsize, self.maxlabellength]) * 10000
        input_length = np.zeros([self.batchsize, 1])
        label_length = np.zeros([self.batchsize, 1])

        print('图片总量', len(_imagefile))
        _imagefile = np.array(_imagefile)
        while 1:
            shufimagefile = _imagefile[self.random_uniform_num(len(_imagefile), self.batchsize)]
            for i, j in enumerate(shufimagefile):
                img_path = os.path.join(self.data_dir, j)
                img1 = Image.open(img_path).convert('L')
                img1 = img1.resize((self.img_w, self.img_h))
                img = np.array(img1, 'f') / 255.0 - 0.5
                x[i] = np.expand_dims(img, axis=2)
                str = image_label[j]
                label_length[i] = len(str)
                if len(str) <= 0:
                    print("len<0", j)
                input_length[i] = imagesize[1] // 8
                labels[i, :len(str)] = [int(i) - 1 for i in str]

            inputs = {'the_input': x,
                      'the_labels': labels,
                      'input_length': input_length,
                      'label_length': label_length,
                      }
            outputs = {'ctc': np.zeros([self.batchsize])}
            yield (inputs, outputs)

    def ctc_lambda_func(self, args):
        labels, y_pred, input_length, label_length = args
        return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(self):
        dense_blstm = denseBlstm()
        base_input = keras.layers.Input(shape=(self.img_h, None, 1), name='the_input')
        base_output = dense_blstm .build_dense_blstm(base_input, self.nclass)
        base_model = keras.models.Model(inputs=base_input, outputs=base_output)
        base_model.summary()

        y_pred = base_output
        labels = keras.layers.Input(name='the_labels', shape=[self.maxlabellength], dtype='float32')
        input_length = keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = keras.layers.Input(name='label_length', shape=[1], dtype='int64')
        output = keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length,label_length])

        self.model = keras.models.Model(inputs=[base_input, labels, input_length, label_length], outputs=output)
        return self.model

    def train(self):
        adam = keras.optimizers.Adam()
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
        checkpoint = keras.callbacks.ModelCheckpoint(self.checkpoint_dir+'/denseblstm_weights.h5',
                                                     monitor='val_loss', save_best_only=True, save_weights_only=True)
        earlystop = keras.callbacks.EarlyStopping(patience=10)
        tensorboard = keras.callbacks.TensorBoard(self.tensorboard_dir, write_graph=True)

        gen_train_data = self.gen(self.trainfile)
        gen_validation_data = self.gen(self.validationfile)

        self.model.fit_generator(gen_train_data,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=gen_validation_data,
                                 validation_steps=self.validation_steps,
                                 callbacks=[earlystop, checkpoint, tensorboard],
                                 verbose=1
                                 )

    def save_model(self):
        pass

    def run(self):
        self.create_dict()
        self.build_model()
        self.train()
        self.save_model()
