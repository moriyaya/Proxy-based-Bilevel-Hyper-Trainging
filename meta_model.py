import numpy as np
import tensorflow as tf
import keras.backend as K


from keras import optimizers
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Subtract, Conv2D, concatenate, subtract
from keras.engine.topology import Layer
from denoise_net import denoise_model, cbd_model, noise_est, ffd_model, small_ffd_model, rela_est

def _get_inputs(img_shape=(None,None,1),kernel_shape=(None,None)):
    x_in = Input(shape=img_shape,    name="x_in")
    y    = Input(shape=img_shape,    name="y")
    k    = Input(shape=kernel_shape, name="k")
    # s    = Input(shape=img_shape,         name="s")
    s = Input(shape=(1,), name="s")
    return x_in, y, k, s

def _get_fuck(img_shape=(None,None,1),kernel_shape=(None,None)):
    x_in = Input(shape=img_shape,    name="x_in")
    y    = Input(shape=img_shape,    name="y")
    k    = Input(shape=kernel_shape, name="k")
    s    = Input(shape=img_shape,    name="s")
    return x_in, y, k, s

def psf2otf(psf, img_shape):
    psf_shape = tf.shape(psf)
    psf_type = psf.dtype

    midH = tf.floor_div(psf_shape[0], 2)
    midW = tf.floor_div(psf_shape[1], 2)

    top_left     = psf[:midH, :midW, :, :]
    top_right    = psf[:midH, midW:, :, :]
    bottom_left  = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]

    zeros_bottom = tf.zeros([psf_shape[0] - midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    zeros_top    = tf.zeros([midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)

    top    = tf.concat([bottom_right, zeros_bottom, bottom_left], 1)
    bottom = tf.concat([top_right,    zeros_top,    top_left],    1)

    zeros_mid = tf.zeros([img_shape[0] - psf_shape[0], img_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    pre_otf = tf.concat([top, zeros_mid, bottom], 0)

    otf = tf.fft2d(tf.cast(tf.transpose(pre_otf, perm=[2,3,0,1]), tf.complex64))

    return otf


class deconvolution_stage(Layer):
    def __init__(self, rho, **kwargs):
        self.rho = rho
        super(deconvolution_stage, self).__init__(**kwargs)

    def build(self, input_shapes):
        # self.rho = K.variable(0.01)
        # self.trainable_weights = [self.rho]
        super(deconvolution_stage, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        x_t, y, k = inputs

        imagesize = tf.shape(x_t)[1:3]
        kk = tf.expand_dims(tf.transpose(k, [1,2,0]), -1)
        fft_k = psf2otf(kk, imagesize)[:,0,:,:]
        denominator = tf.square(tf.abs(fft_k))
        fft_y = tf.fft2d(tf.cast(y[:,:,:,0], tf.complex64))

        upperleft = tf.conj(fft_k) * fft_y

        fft_x = tf.fft2d(tf.cast(x_t[:,:,:,0], tf.complex64))

        # rho = tf.squeeze(tf.cast(rho, tf.float64), axis=-1)
        # rho = tf.fft2d(tf.cast(rho, tf.complex64))
        # z = (upperleft + rho*fft_x) / (tf.cast(denominator, tf.complex64) + rho)

        rho = tf.cast(tf.expand_dims(self.rho, -1), tf.float64)
        # rho = tf.cast(rho, tf.float64)
        z = (upperleft + tf.cast(rho, tf.complex64)*fft_x) / tf.cast((tf.cast(denominator, tf.float64) + rho), tf.complex64)
        z = tf.to_float(tf.ifft2d(z))
        z1 = tf.expand_dims(z, -1)

        return z1

class denoise_stage(Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.net = denoise_model()
        super(denoise_stage, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.trainable_weights = self.net.get_weights()
        super(denoise_stage, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, inputs):
        x = inputs
        # residual = self.model(x)

        # net = denoise_model()
        self.net.load_weights('./models/ircnn/net1.hdf5')
        residual = self.net(x)

        z = x - residual
        return z

class fusion_stage(Layer):
    def __init__(self, **kwargs):
        super(fusion_stage, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(fusion_stage, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        x_out, x_denoise, lamb = inputs
        # lamb = K.expand_dims(lamb, -1)
        # lamb = K.expand_dims(lamb, -1)
        z = lamb*x_out + (1-lamb)*x_denoise
        return z

def meta_stage(rho):
    x_t, y, k, s = _get_fuck()
    x_out = deconvolution_stage(rho)([x_t, y, k])

    denoiser = cbd_model()
    # denoiser = ffd_model()
    # denoiser.load_weights('./models/cbd/best.hdf5')
    x_out = denoiser(x_out)

    # related_net = noise_est()
    # related = related_net(x_out)

    # x_out = fusion_stage()([x_out, x_denoise, related])
    return Model([x_t, y, k], x_out)

def meta_stack(stage, rho_idx, weights=[]):
    x, y, k, s = _get_fuck()
    output = []
    for i in range(stage):
        model = meta_stage(rho_idx[i])
        if len(weights) > 0:
            model.load_weights(weights[i])
        output.append(model([(output[-1] if i>0 else x), y, k]))

    return Model([x, y, k], output[-1])


def meta_stage_divide(divide=False, rho=0.01):
    if divide:
        x_t, y, k, s = _get_fuck()
        x_out = deconvolution_stage(rho)([x_t, y, k])
        related_net = noise_est()
        related = related_net(x_out)

        net2_input = concatenate([x_out, related], axis=-1)
        net2 = ffd_model(False)
        noise = net2(net2_input)
        denoise = subtract([x_out, noise])
        m2 = Model([x_t, y, k], [x_out, related, denoise])
        # m2.load_weights('./models/denoise.hdf5')
        # m1 = Model([x_t, y, k], related)
        m1 = Model([x_t, y, k], related)
        return m1, m2

    else :
        x_t, y, k, s = _get_fuck()
        x_out = deconvolution_stage(rho)([x_t, y, k])
        related_net = noise_est()
        related = related_net(x_out)
        net2_input = concatenate([x_out, related], axis=-1)
        net2 = ffd_model(False)
        noise = net2(net2_input)
        denoise = subtract([x_out, noise])
        m = Model([x_t, y, k], [x_out, related, denoise])
        # m.load_weights('./models/denoise.hdf5')
        return m

def meta_stack_divide(stage, rho_index):
    x, y, k, s = _get_fuck()
    output = []
    weights = './models/cbd/deblur/divide/stage_best.hdf5'

    m1, m2 = meta_stage_divide(True, rho_index[0])
    m1.compile(loss='mse', optimizer=optimizers.Adam(1e-4), loss_weights=[100])
    # m1.trainable = False
    # m2 = meta_stage_divide(False, rho_index[0])
    temp = m2([x, y, k])
    output.append(temp[0])
    output.append(temp[1])
    output.append(temp[2])
    # output.append(m2([x, y, k]))

    for i in range(1, stage):
        model = meta_stage_divide(False, rho_index[i])
        # model.load_weights(weights)
        # if len(weights) > 0:
        #     model.load_weights(weights[i])
        temp = model([(output[-1] if i>0 else x), y, k])
        output.append(temp[0])
        output.append(temp[1])
        output.append(temp[2])
    
    m2 = Model([x, y, k], output)
    m2.compile(loss='mse', optimizer=optimizers.Adam(1e-4))
    return m1, m2


def meta_nohyper_stage(rho=0.01):
    x_t, y, k, s = _get_fuck()
    x_out = deconvolution_stage(rho)([x_t, y, k])
    # net2 = ffd_model(False, 1)
    net2 = small_ffd_model(False, 1)
    noise = net2(x_out)
    denoise = subtract([x_out, noise])
    m = Model([x_t, y, k], denoise)
    # m.load_weights('./models/denoise.hdf5')
    return m

def meta_nohyper_stack(stage, rho_index):
    x, y, k, s = _get_fuck()
    output = []

    m2 = meta_nohyper_stage(rho_index[0])
    output.append(m2([x, y, k]))

    for i in range(1, stage):
        model = meta_nohyper_stage(rho_index[i])
        output.append(model([(output[-1] if i>0 else x), y, k]))
    
    m2 = Model([x, y, k], output[-1])
    m2.compile(loss='mse', optimizer=optimizers.Adam(1e-4))
    return m2

def meta_man_stage(rho=0.01, fuck=False):
    x_t, y, k, s = _get_fuck()
    x_out = deconvolution_stage(rho)([x_t, y, k])

    if fuck:
        related_net = noise_est()
        related = related_net(x_out)
        x_in = concatenate([x_out, related], axis=-1)
    else :
        x_in = concatenate([x_out, s], axis=-1)
    # net2 = ffd_model(False)
    net2 = small_ffd_model(False)
    noise = net2(x_in)
    denoise = subtract([x_out, noise])
    m = Model([x_t, y, k, s], denoise)
    # m.load_weights('./models/denoise.hdf5')
    return m

def meta_man_stack(stage, rho_index):
    x, y, k, s = _get_fuck()
    output = []

    m2 = meta_man_stage(rho_index[0])
    output.append(m2([x, y, k, s]))

    for i in range(1, stage):
        model = meta_man_stage(rho_index[i], fuck=True)
        output.append(model([(output[-1] if i>0 else x), y, k, s]))
    
    m2 = Model([x, y, k, s], output[-1])
    m2.compile(loss='mse', optimizer=optimizers.Adam(1e-4))
    return m2

if __name__ == '__main__':
    # m1, m2 = meta_stage_divide(True)
    # m1.summary()
    # m2.summary()

    # m = meta_nohyper_stack(5, range(5))
    m = meta_man_stack(5, range(5))
    m.summary()
