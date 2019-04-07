# -*- coding: utf-8 -*-

from form_recognition import formRecognition
from OCR import OCR
from preprocessor import select_imgs_with_form, correct_img_bias
from PIL import Image
from pdf2img import pdf2img
import cv2
import argparse
import logging
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_file", default="",
                    help="input PDF path")

    args = vars(ap.parse_args())
    return args


def main(args):
    pdf_file = args['pdf_file']
    pdf_images = pdf2img(pdf_file)
    images = pdf_images
    # images = select_imgs_with_form(pdf_images)
    ocr = OCR()
    for index, image in enumerate(images):
        image = correct_img_bias(image)
        fr = formRecognition(image)
        tables = fr.run()
        for i in range(len(tables)):
            table = tables[i]
            for idx, cell in enumerate(table):
                cell_img = ocr.img_enhancement(cell)
                res = ocr.text_recognition(cell_img)
                print(res, end='\n')
            #Todo: 按照原表格的结构，重新组织识别出的单元格内容，导出电子表格


if __name__ == "__main__":
    args = args_parse()
    main(args)
