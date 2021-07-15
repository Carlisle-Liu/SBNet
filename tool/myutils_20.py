
import PIL.Image
import random
import numpy as np
from skimage import transform

class RandomResizeLong():
    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, input, sal=None):

        target_long = random.randint(self.min_long, self.max_long)
        img = input[0]
        npy = input[1]
        w, h = img.size

        if w < h:
            target_shape_img = (int(round(w * target_long / h)), target_long)
            target_shape_npy = (21, target_long, int(round(w * target_long / h)))
        else:
            target_shape_img = (target_long, int(round(h * target_long / w)))
            target_shape_npy = (21, int(round(h * target_long / w)), target_long)

        # print("w: ", w, "h: ", h, "Img: ", "target: ", target_long, target_shape_img, "npy: ", target_shape_npy)

        img = img.resize(target_shape_img, resample=PIL.Image.CUBIC)
        npy = transform.resize(npy, target_shape_npy, order=3, mode='edge')

        if sal:
           sal = sal.resize(target_shape, resample=PIL.Image.CUBIC)
           return img, sal
        return img, npy


class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, input, sal=None):
        imgarr = input[0]
        npy = input[1]
        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container_img = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container_img[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
            imgarr[img_top:img_top + ch, img_left:img_left + cw]

        container_npy = np.zeros((npy.shape[0], self.cropsize, self.cropsize), np.float32)
        container_npy[:, cont_top:cont_top + ch, cont_left:cont_left + cw] = \
            npy[:, img_top:img_top + ch, img_left:img_left + cw]
        if sal is not None:
            container_sal = np.zeros((self.cropsize, self.cropsize, 1), np.float32)
            container_sal[cont_top:cont_top + ch, cont_left:cont_left + cw, 0] = \
                sal[img_top:img_top + ch, img_left:img_left + cw]
            return container, container_sal
        return container_img, container_npy


class RandomHorizontalFlip():
    def __call__(self, input):
        img = input[0]
        npy = input[1]

        if random.random() > 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for i in range(npy.shape[0]):
                npy[i,:,:] = np.flip(npy[i,:,:], axis=1)

        return img, npy

class RandomHorizontalFlip_Npy(object):
    def __call__(self, img):

        do_flip = np.random.random() > 0.5

        if do_flip:
            img = np.flip(img, axis=1)

        return img
