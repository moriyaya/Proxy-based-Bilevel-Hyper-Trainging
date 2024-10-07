import os
import copy
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from utils import comp_quantitative as PSNR
from scipy.signal import fftconvolve
from keras.backend.tensorflow_backend import set_session

from meta_model import meta_stack, meta_stage
from denoise_net import denoise_model
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.measure import compare_psnr, compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def get_rho_and_net_index(sigma, iter_num):
    rho_index=np.zeros([iter_num,])
    net_index=np.zeros([iter_num,])

    lambda_=(sigma**2)/3
    modelSigma1=49
    modelSigma2=13
    modelSigmaS=np.logspace(np.log10(modelSigma1),np.log10(modelSigma2),iter_num)
    for i in range(iter_num):
        rho_index[i]=(lambda_*255**2)/(modelSigmaS[i]**2)
    
    net_index=np.ceil(modelSigmaS/2)
    net_index=np.clip(net_index,1,25)

    return rho_index, net_index

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]
    elif img.ndim == 4:
        return img

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def pad_for_kernel(img,kernel,mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p,p] + (img.ndim-2)*[(0,0)]
    return np.pad(img, padding, mode)

def crop_for_kernel(img,kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0],-p[0]),slice(p[1],-p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]

def edgetaper_alpha(kernel,img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel,1-i),img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z,z[0:1]],0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)

# use edge processing if necessary
def edgetaper(img,kernel,n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[...,np.newaxis]
        alpha  = alpha[...,np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img,_kernel,'wrap'),kernel,mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def test_level(model):
    im_list = ['05','06','07','08']
    sigmas = [0.01, 0.02, 0.03, 0.04]
    psnrsum = 0
    ssimsum = 0

    for im_idx in im_list:
        for sigma in sigmas:
            name = "./Levin09blurdata/im" + im_idx +"_flit01.mat"
            data = scipy.io.loadmat(name)
            kernel  = data['f']
            gt      = data['x']
            blurred = gt + sigma*np.random.standard_normal(size=gt.shape)

            y = to_tensor(blurred)
            x0 = y
            k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))

            ss = np.full((1,y.shape[1],y.shape[2],1), sigma)
            pred = model.predict_on_batch([x0, y, k])
            predi = from_tensor(pred)

            gt = sigma * np.ones(gt.shape)
            gt = gt.astype(np.float64)
            predi = predi.astype(np.float64)
            psnr = compare_psnr(gt, predi)
            psnrsum += psnr

    return psnrsum / (len(im_list) * len(sigmas))


def testsun(model, sigma):

    ke_list = ['01','02','03','04','05','06','07','08']
    psnrsum = 0
    ssimsum = 0

    for im_idx in range(1, 81):
        for ke_idx in ke_list:
            # print(str(im_idx) + ' ' + str(ke_idx))
            name = "./Levin09blurdata/im05_flit" + ke_idx + ".mat"
            data = scipy.io.loadmat(name)
            kernel  = data['f']
            gt = img_as_float(imread('./../../data/deblur/sun/gt/img%d_groundtruth_img.png'%im_idx))
            sigma = np.ones(gt.shape) * 0.07
            blurred = fftconvolve(gt, kernel, mode='same') + sigma*np.random.standard_normal(size=gt.shape)

            y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
            x0 = y
            k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))

            pred = model.predict_on_batch([x0, y, k])
            predi = crop_for_kernel(from_tensor(pred),kernel)

            psnr, ssim = PSNR(gt, predi, kernel.shape[0])
            print("%d %s %f %f"%(im_idx, ke_idx, psnr, ssim))
            # imsave('./sun/%d_%d.png'%(im_idx, int(ke_idx[1])), np.clip(predi, 0, 1))
            psnrsum += psnr
            ssimsum += ssim

    print(psnrsum / 640)
    print(ssimsum / 640)

def test_pic(model):
    gt = img_as_float(imread('./data/noise_level_est/gt/gt.JPEG'))
    rgbblurred = copy.deepcopy(gt)
    name = "./Levin09blurdata/im05_flit07.mat"
    data = scipy.io.loadmat(name)
    kernel  = data['f']
    sigma = np.ones(gt.shape[:2]) * 0.015
    # sigma = 0.03
    # sigma[0 : gt.shape[0]/2, 0 : gt.shape[1]/2] = 0.01
    # sigma[0 : gt.shape[0]/2, gt.shape[1]/2 : gt.shape[1]] = 0.02
    # sigma[gt.shape[0]/2 : gt.shape[0], 0 : gt.shape[1]/2] = 0.03
    # sigma[gt.shape[0]/2 : gt.shape[0], gt.shape[1]/2 : gt.shape[1]] = 0.04
    # sigma[:, 0 : gt.shape[1]/2] = 0.01
    # sigma[:, gt.shape[1]/2 : gt.shape[1]] = 0.02
    # for i in range(gt.shape[2]):
        # rgbblurred[:,:,i] = fftconvolve(gt[:,:,i], kernel, mode='same') + sigma*np.random.standard_normal(size=gt.shape[:2])
    # imsave('blurrgb.png', np.clip(rgbblurred, 0, 1))

    gt = rgb2gray(gt)
    blurred = fftconvolve(gt, kernel, mode='same') + sigma*np.random.standard_normal(size=gt.shape[:2])
    # imsave('blur.png', np.clip(blurred, 0, 1))

    y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
    x0 = y
    k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))

    weight = model.predict_on_batch([x0, y, k])
    weight = crop_for_kernel(from_tensor(weight),kernel)
    # np.save('weight1234.npy',weight)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    im1 = ax1.imshow(imread("./data/noise_level_est/1_5/blurrgb.png"))
    im2 = ax2.imshow(sigma, cmap=plt.cm.spring, vmin=0, vmax=0.025)
    im3 = ax3.imshow(weight, cmap=plt.cm.spring, vmin=0, vmax=0.025)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85 ,0.38, 0.05,0.24])
    plt.colorbar(im3, cax=cbar_ax)
    # plt.colorbar()
    # plt.show()
    plt.savefig('./data/noise_level_est/level_1_5.pdf')
    

def testsun_color(model):
    for i in range(1, 81):
        for j in range(1, 9):
            img_ke.append((i, j))

    for _sigma in range(1, 6):
        for im_idx, ke_idx in img_ke:
            name = "./Levin09blurdata/im05_flit0" + str(ke_idx) + ".mat"
            data = scipy.io.loadmat(name)
            kernel  = data['f']
            sigma = _sigma / 100.0
            gt_color = img_as_float(imread('./../../data/deblur/sun/gt_color/%d.jpg'%im_idx))
            ans_color = img_as_float(imread('./../../data/deblur/sun/gt_color/%d.jpg'%im_idx))
            for i in range(3):
                gt = gt_color[:,:,i]
                blurred = fftconvolve(gt, kernel, mode='same') + sigma*np.random.standard_normal(size=gt.shape)
                y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
                x0 = y
                k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))

                pred = model.predict_on_batch([x0, y, k])
                predi = crop_for_kernel(from_tensor(pred),kernel)
                ans_color[:,:,i] = predi

            # imsave('./sun/%d_%d_%d.png'%(im_idx, ke_idx, _sigma), np.clip(ans_color, 0, 1))
            imsave('/data/sun/ours/%d_%d_%d.png'%(im_idx, ke_idx, _sigma), np.clip(ans_color, 0, 1))

def testlevin(model, sigma, need_sigma=False):

    im_list = ['05','06','07','08']
    ke_list = ['01','02','03','04','05','06','07','08']
    im_list = ['07']
    ke_list = ['04']
    psnrsum = 0
    ssimsum = 0

    for im_idx in im_list:
        for ke_idx in ke_list:
            name = "./Levin09blurdata/im" + im_idx +"_flit" + ke_idx + ".mat"
            data = scipy.io.loadmat(name)
            # blurred = data['y']
            kernel  = data['f']
            # kernel = np.rot90(kernel,2)
            gt      = data['x']
            sigmas = np.ones(gt.shape) * sigma


            blurred = fftconvolve(gt, kernel, mode='same') + sigmas*np.random.standard_normal(size=gt.shape)

            y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
            x0 = y
            k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))
            # sigmas = np.tile(sigma, (y.shape[0], 1)).astype(np.float32)
            imsave('./data/levin/kernel.png', np.clip(kernel*255, 0, 1))
            imsave('./data/levin/blur.png', np.clip(blurred, 0, 1))

            ss = np.full((1,y.shape[1],y.shape[2],1), sigma)
            if not need_sigma:
                pred = model.predict_on_batch([x0, y, k])
            else :
                pred = model.predict_on_batch([x0, y, k, ss])
            pred = pred[-1]
            predi = crop_for_kernel(from_tensor(pred),kernel)


            psnr, ssim = PSNR(gt, predi, kernel.shape[0])
            print('im:%s ke:%s PSNR:%f'%(im_idx, ke_idx, psnr))
            psnrsum += psnr
            ssimsum += ssim

    print(psnrsum / (len(im_list) * len(ke_list)))
    print(ssimsum / (len(im_list) * len(ke_list)))


def testreal(model):

    for i in range(41):
        kernel  = img_as_float(imread('./../../data/deblur/real/kernel/'+str(i)+'.png'))
        kernel /= np.sum(kernel)
        blurred = img_as_float(imread('./../../data/deblur/real/blur/'+str(i)+'.png'))

        y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
        x0 = y
        k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))

        pred = model.predict_on_batch([x0, y, k])
        predi = crop_for_kernel(from_tensor(pred),kernel)

        imsave('./out/real/'+str(i)+'.png', np.clip(predi, 0, 1))



if __name__=="__main__":

    K.clear_session()

    iter_num = 10
    sigma = 0.01
    finetune = True

    rho_index, net_index = get_rho_and_net_index(sigma, iter_num)
    print(rho_index)
    sigma = 0.05
    test(sigma, rho_index)

