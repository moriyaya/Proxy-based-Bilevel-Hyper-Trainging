import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.io import imread
from scipy.signal import fftconvolve
from utils import edgetaper, pad_for_kernel


class DataLoader:

    def __init__(self, rootpath='./../../remote/BSDS300/', batch_size=5):
        self.rotpath = rootpath
        self.crop_size = 320
        self.batch_size = batch_size
        self.k_sz_max = (37, 37)
        self.images = []
        imageslist = os.listdir(rootpath)
        for i in xrange(len(imageslist)):
            if imageslist[i].find('.jpg') != -1:
                self.images.append(imageslist[i])

    def get_batch(self):
        x_in = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1), dtype=np.float32)
        y_in = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1), dtype=np.float32)
        k_in = np.zeros((self.batch_size, self.k_sz_max[0], self.k_sz_max[1]), dtype=np.float32)
        s_in = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1), dtype=np.float32)
        # s_in = np.zeros((self.batch_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            index = np.random.randint(0, len(self.images))
            x = img_as_float(imread(os.path.join(self.rotpath, self.images[index]), True)).astype(np.float32)  # gray scale
            w, h = x.shape
            pos_x = np.random.randint(0, w - self.crop_size)  # half open range
            pos_y = np.random.randint(0, h - self.crop_size)
            x = x[pos_x:pos_x + self.crop_size, pos_y:pos_y + self.crop_size]

            k_index = np.random.randint(1, 21)
            k = np.loadtxt(os.path.join('./../../remote/kernel28', 'kernel_%02d.dlm' % k_index))
            k = np.clip(k, 0, 1)
            k /= np.sum(k)
            if i%2==0:
                k = k[::-1,::-1] # flip kernel

            y = fftconvolve(x, k, mode='valid')
            s = np.random.randint(100, 500)/100.0  # rand 1-3
            # y = y + (s / 255.0) * np.random.standard_normal(size=y.shape)
            y = y + (s / 100.0) * np.random.standard_normal(size=y.shape)
            y = np.clip(y, 0, 1.0)
            y = ((y * 255.0).astype(np.uint8) / 255.0).astype(np.float32)
            y = edgetaper(pad_for_kernel(y, k, 'edge'), k)

            if self.k_sz_max != k.shape:
                excess = [(self.k_sz_max[d] - k.shape[d]) // 2 for d in range(k.ndim)]
                k = np.pad(k, excess, 'constant')   # pad kernel with zeros for the same size

            x_in[i, :, :, 0] = x
            y_in[i, :, :, 0] = y
            k_in[i, :, :] = k
            s_in[i, :, :, 0] = np.full((x.shape[0], x.shape[1]), s/100.0)
            # s_in[i, 0] = s

        return x_in, y_in, k_in, s_in


class denoiseDataLoader:
    def __init__(self, rootpath='./../../remote/BSDS300/', batch_size=5):
        self.rotpath = rootpath
        self.crop_size = 320
        self.batch_size = batch_size
        self.images = []
        imageslist = os.listdir(rootpath)
        for i in xrange(len(imageslist)):
            if imageslist[i].find('.jpg') != -1:
                self.images.append(imageslist[i])

    def get_batch(self, given_sigma=None):
        clear_in = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1), dtype=np.float32)
        noise_in = np.zeros((self.batch_size, self.crop_size, self.crop_size, 2), dtype=np.float32)
        for i in range(self.batch_size):
            index = np.random.randint(0, len(self.images))
            clear = img_as_float(imread(os.path.join(self.rotpath, self.images[index]), True)).astype(np.float32)  # gray scale
            w, h = clear.shape
            pos_x = np.random.randint(0, w - self.crop_size)  # half open range
            pos_y = np.random.randint(0, h - self.crop_size)
            clear = clear[pos_x:pos_x + self.crop_size, pos_y:pos_y + self.crop_size]

            sigmas = range(5, 51, 5)
            s = sigmas[np.random.randint(len(sigmas))] if given_sigma is None else given_sigma
            noise = clear + (s / 255.0) * np.random.standard_normal(size=clear.shape)
            noise = np.clip(noise, 0, 1.0)
            noise = ((noise * 255.0).astype(np.uint8) / 255.0).astype(np.float32)


            clear_in[i, :, :, 0] = clear
            noise_in[i, :, :, 0] = noise
            noise_in[i, :, :, 1] = np.full((clear.shape[0], clear.shape[1]), s/255.0)

        return clear_in, noise_in

    def get_all_sigma_batch(self):
        sigmas = range(5, 51, 5)
        batch = len(sigmas)
        crop_size = 160
        clear_in = np.zeros((batch, crop_size, crop_size, 1), dtype=np.float32)
        noise_in = np.zeros((batch, crop_size, crop_size, 2), dtype=np.float32)

        for i in range(batch):
            index = np.random.randint(0, len(self.images))
            clear = img_as_float(imread(os.path.join(self.rotpath, self.images[index]), True)).astype(np.float32)  # gray scale
            w, h = clear.shape
            pos_x = np.random.randint(0, w - crop_size)  # half open range
            pos_y = np.random.randint(0, h - crop_size)
            clear = clear[pos_x:pos_x + crop_size, pos_y:pos_y + crop_size]

            sigmas = range(5, 51, 5)
            s = sigmas[i]
            noise = clear + (s / 255.0) * np.random.standard_normal(size=clear.shape)
            noise = np.clip(noise, 0, 1.0)
            noise = ((noise * 255.0).astype(np.uint8) / 255.0).astype(np.float32)

            clear_in[i, :, :, 0] = clear
            noise_in[i, :, :, 0] = noise
            noise_in[i, :, :, 1] = np.full((clear.shape[0], clear.shape[1]), s/255.0)

        return clear_in, noise_in

if __name__ == '__main__':
    # data = denoiseDataLoader(batch_size=1)
    # clear, noise = data.get_batch(10)
    # plt.subplot(1,2,1)
    # plt.imshow(clear[0,:,:,0], 'gray')
    # plt.subplot(1,2,2)
    # plt.imshow(noise[0,:,:,0], 'gray')
    # print(noise[0,:,:,1])
    # plt.show()
 
    data = DataLoader(batch_size=1)
    x, y, k, s = data.get_batch()
    plt.subplot(1,2,1)
    plt.imshow(x[0,:,:,0], 'gray')
    plt.subplot(1,2,2)
    plt.imshow(y[0,:,:,0], 'gray')
    print(s[0,:,:,0])
    plt.show()
 
