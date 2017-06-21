# part I & II of the tutorials here: https://prateekvjoshi.com/2016/02/02/deep-learning-with-caffe-in-python-part-i-defining-a-layer/

import sys
sys.path.insert(0, '../caffe/python')
import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb2gray(rgb):
# to turn color images to grey
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

net = caffe.Net(str("nets/my_train_val.prototxt"), str("my_train_val_iter_170.caffemodel"))

# load image into the first layer
im = np.array(Image.open('jpgs/cfgRank1.jpg'))
color = False
if color == True:
    im_input = im[np.newaxis, np.newaxis, :, :]
    net.blobs['data'].reshape(1, 3, 256, 256)
    net.blobs['data'].data[...] = im_input
else:
    im = rgb2gray(im)
    im_input = im[np.newaxis, np.newaxis, :, :]
    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input

net.forward()

for i in range(10):
    plt.figure(i)
    plt.imshow(net.blobs['conv'].data[0,i])
plt.show()