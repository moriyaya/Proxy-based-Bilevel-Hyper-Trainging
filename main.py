from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import os, sys, json, argparse, datetime
import keras.backend as K
import keras
import skimage
import time

from scipy.signal import fftconvolve
from skimage.io import imread, imsave
from skimage import img_as_float
from pprint import pprint
from meta_model import meta_stack_divide, meta_stage_divide, meta_stack, meta_nohyper_stack, meta_man_stack
from dataloader import DataLoader
from denoise_net import denoise_model
from utils import get_rho_and_net_index

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras import optimizers
from keras.layers import Input
from keras.models import Model
from test import testsun, testsun_color, testlevin, testreal, test_pic, test_level

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto(device_count={'gpu':0})  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

KTF.set_session(sess)

def psnr_loss(y_true, y_pred):
    peakval = 1.
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return -10. *(K.log( (peakval * peakval) / (mse + K.epsilon()) ) / K.log(10.))

def finetune_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def _train_divide(model1, model2, itr_max, batch_size=1,  lr=0.0001, weights_root=None):
    data_loader = DataLoader(batch_size=batch_size)
    # model1.compile(loss=finetune_loss, optimizer=optimizers.Adam(lr))
    itr = 1
    while itr <= itr_max:
        x, y, k, s = data_loader.get_batch()
        x0 = y
        xx = []
        los1 = model1.train_on_batch([x0, y, k], s)
        los2 = model2.train_on_batch([x0, y, k], x)
        
        print((itr, los1, los2))
        # if itr % 10000 == 1:
        #     model2.save_weights('%s/%d.hdf5'%(weights_root, itr))
        itr+=1

def _train(model, itr_max, batch_size=1,  lr=0.0001, weights_root=None):
    data_loader = DataLoader(batch_size=batch_size)
    # model1.compile(loss=finetune_loss, optimizer=optimizers.Adam(lr))
    itr = 1
    while itr <= itr_max:
        x, y, k, s = data_loader.get_batch()
        x0 = y
        xx = []
        los = model.train_on_batch([x0, y, k, s], x)
        
        print((itr, los))
        if itr % 10000 == 1:
            model.save_weights('%s/%d.hdf5'%(weights_root, itr))
        itr+=1

if __name__ == "__main__":
    K.clear_session()
    n_stages = 5
    epoch = 100010
    batch = 3
    weights_root='models/manual_hyper'

    rho_index, _ = get_rho_and_net_index(1e-2, n_stages)
    rho_index = np.array(rho_index).astype(np.float64)
    
    K.clear_session()
    # m1, m2 = meta_stage_divide()
    m1, m2 = meta_stack_divide(5, rho_index)
    
    # train
    # _train(m, epoch, batch, lr=1e-4, weights_root=weights_root)
    # _train(m1, m2, epoch, batch, lr=1e-4, weights_root=weights_root)
    # m.save_weights('%s/stages_01-%02d_finetuned.hdf5'%(weights_root, n_stages))


    # test
    # m2.load_weights('models/combine_best.hdf5')
    m2.load_weights('models/200001.hdf5')

    # test_level(m1)
    # test_pic(m1)
    testlevin(m2, 0.01, False)
    # testreal(m2)
    # testsun(m2, 0.05)
    # testsun_color(m2)


