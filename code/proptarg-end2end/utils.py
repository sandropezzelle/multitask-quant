import skimage
import skimage.io
import skimage.transform
import numpy as np


'''
function implemented in the tensorflow-vgg for reading images
'''
def load_image2(path, dim):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1).all()
    print "original image shape :", img.shape
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    print "cropped image :", crop_img.shape
    resized_img = skimage.transform.resize(crop_img, (dim, dim))
    print "resized image :", resized_img.shape
    return resized_img

def load_image(path, dim):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1).all()
#    print "1original image shape :", img.shape
    resized_img = skimage.transform.resize(img, (dim, dim))
#    print "1resized image :",resized_img.shape
    return resized_img


def read_qprobs(path):
    fin = open(path + '/code/Q-probabilities.txt', 'r')
    count = 0
    ratios = {}
    ratio_l = []
    for line in fin:
        els = line.strip().split('\t')
        if count == 0:
            for el in els[1:]:
                ratios[el] = {}
            ratio_l = els[1:]
            for el in ratios:
                for i in range(9):
                    ratios[el][str(i)] = 0.0
        else:
            ind = els[0]
            for i in range(17):
                val = els[2 + i]
                r = ratio_l[i]
                ratios[r][ind] = val
        count += 1
    return ratios


if __name__ == '__main__':
    print read_qprobs()
