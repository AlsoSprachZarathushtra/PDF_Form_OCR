# -*- coding: utf-8 -*-

#=======================   train   ========================================
# config = {
#     'char_dict_path': '/home/zhangjiacheng/ai_image/table/chinese_recognition/crnn_ocr/dict/char_std_5990.txt',
#     'model_path': '/home/zhangjiacheng/data/table/chinese_ocr/checkpoints/denseblstm_weights.h5',
#     'trainfile': '/home/zhangjiacheng/data/table/chinese_ocr/data/Synthetic/data_train.txt',
#     'validationfile': '/home/zhangjiacheng/data/table/chinese_ocr/data/Synthetic/data_test.txt',
#     'data_dir': '/home/zhangjiacheng/data/table/chinese_ocr/data/Synthetic/images',
#     'checkpoint_dir': '/home/zhangjiacheng/data/table/chinese_ocr/checkpoints',
#     'tensorboard_dir': '/home/zhangjiacheng/data/table/chinese_ocr/tflog-densent',
#     'img_h': 32,
#     'img_w': 280,
#     'maxlabellength': 10,
#     'batchsize': 4,
#     'epochs': 3,
#     'steps_per_epoch': 100,
#     'validation_steps': 5
# }
#
# crnn_trainer = crnnTrainer(config)
# crnn_trainer.run()

#=======================   predict   ========================================
from predict import crnnPredictor
import cv2
config = {
    'char_dict_path': '/home/zhangjiacheng/ai_image/table/form_recognition/src/crnn_ocr/dict/char_std_5990.txt',
    'model_path': '/home/zhangjiacheng/data/table/chinese_ocr/checkpoints/denseblstm_weights.h5',
    'img_h': 32,
    'img_w': 280,
}
crnn_predictor = crnnPredictor(config)
img_path = '/home/zhangjiacheng/data/table/chinese_ocr/test_imgs/hangzhou_imgs/cell_1.jpg'
img = cv2.imread(img_path)
crnn_predictor.run(img)


