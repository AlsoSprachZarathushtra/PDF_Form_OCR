from pdf2image import convert_from_path
import tempfile
import numpy as np
from PIL import Image

def pdf2img(filename):
    print('filename=', filename)
    # print('outputDir=', outputDir)
    imgs = []
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(filename)
        for i in range(len(images)):
            image = np.array(images[i])
            image = Image.fromarray(image)
            imgs.append(image)

        # for index, img in enumerate(images):
        #     img.save('%s/page_%s.png' % (outputDir, index))
    return imgs


if __name__ == "__main__":
    pdf2img('/home/zhangjiacheng/data/form_recognition/new/hang.pdf', '/home/zhangjiacheng/data/form_recognition/new/data/hang')

