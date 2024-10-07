import numpy as np

from scipy.signal import fftconvolve
from skimage.measure import compare_ssim, compare_psnr

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

def psnr(x,y):
    return -10 * np.math.log10(((x -  y)** 2).mean())

def comp_quantitative(fe, sharp_ground,k):
    fe = fe.astype('float64')
    sharp_ground = sharp_ground.astype('float64')

    ks =int(np.floor(k/2))
    fe1 = fe[0 + ks: fe.shape[-2] - ks , 0 + ks : fe.shape[-1] - ks]
    m = fe1.shape[-2]
    n = fe1.shape[-1]
    psnr0 = 0.0
    ssim0 = 0.0
    m1 = sharp_ground.shape[0]
    n1 = sharp_ground.shape[1]
    for i in range(0,m1-m ,1):
        for j in range(0,n1-n ,1):
            sharp_ground1 = sharp_ground[i:m+i,j:j+n]
            psnr1 = psnr(fe1,sharp_ground1)
            # psnr1 = compare_psnr(sharp_ground1, fe1)
            if psnr1 > psnr0:
                psnr0 = psnr1
                ssim1 = compare_ssim(fe1, sharp_ground1)
                ssim0 = max(ssim0, ssim1)
            # ssim1 = 0
            # psnr0 = max(psnr0,psnr1)
            # ssim0 = max(ssim0,ssim1)
    return psnr0, ssim0


def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]


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


