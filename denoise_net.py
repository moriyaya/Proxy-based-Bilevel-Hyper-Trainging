# IRCNN
import keras.backend as K
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, concatenate, Subtract, subtract, GlobalAveragePooling2D, Dense

class mergeLayer(Layer):
    def __init__(self, **kwargs):
        super(mergeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        x, y = inputs
        return x - y


def denoise_model(bn=False):
    image_shape=(None,None,1)
    act='relu'
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1,input_shape=image_shape))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=4))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(1,(3,3),padding='same',dilation_rate=1))

    return model

def noise_est():
    image_shape = (None, None, 1)
    act = 'relu'
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1,input_shape=image_shape))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1))
    model.add(Activation(act))
    model.add(Conv2D(1,(3,3),padding='same',dilation_rate=1))
    model.add(Activation(act))
    return model

# def rela_est():
#     image_shape=(None, None, 1)
#     model = Sequential()
#     model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', input_shape=image_shape))
#     model.add(Conv2D(32, (3,3), strides=(2,2), padding='same'))
#     model.add(Conv2D(32, (3,3), strides=(2,2), padding='same'))
#     model.add(Conv2D(1, (3,3), strides=(2,2), padding='same'))
#     model.add(GlobalAveragePooling2D())
#     return model

def rela_est():
    image_shape=(1, )
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=image_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

def ffd_model(bn=True, input_channel=2):
    image_shape=(None,None,input_channel)
    act='relu'
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1,input_shape=image_shape))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))

    for i in range(3):
        model.add(Conv2D(64,(3,3),padding='same',dilation_rate=4))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation(act))

    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(1,(3,3),padding='same',dilation_rate=1))

    return model

def small_ffd_model(bn=True, input_channel=2):
    image_shape=(None,None,input_channel)
    act='relu'
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1,input_shape=image_shape))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))

    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(1,(3,3),padding='same',dilation_rate=1))

    return model

def cbd_model(is_train=False):
    if is_train:
        x_in = Input(shape=(None, None, 1))
        net1 = noise_est()
        def custom_loss(y_true, y_pred):
            def tvloss(y):
                diff1 = y[1:, :, :] - y[:-1, :, :]
                diff2 = y[:, 1:, :] - y[:, :-1, :]
                return K.mean(K.square(diff1)) + K.mean(K.square(diff2))
            def finetune_loss(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true))
            def loss(y_true, y_pred):
                return finetune_loss(y_true, y_pred) + 0.1 * tvloss(y_pred)
            return loss(y_true, y_pred)
        net1.compile(loss=custom_loss, optimizer='adam')

        net1.trainable = False
        noise_map = net1(x_in)
        net2_input = concatenate([x_in, noise_map], axis=-1)
        net2 = ffd_model(False)
        noise = net2(net2_input)
        denoise = subtract([x_in, noise])
        net_whole = Model(x_in, denoise)
        net_whole.compile(loss='mse', optimizer='adam')
        return net1, net_whole
    else :  
        x_in = Input(shape=(None, None, 1))
        net1 = noise_est()
        noise_map = net1(x_in)
        net2_input = concatenate([x_in, noise_map], axis=-1)
        net2 = ffd_model(False)
        noise = net2(net2_input)
        denoise = subtract([x_in, noise])
        net_whole = Model(x_in, denoise)
        return net_whole


# def cbd_model(is_train=False):
#     x_in = Input(shape=(None, None, 1))
#     net1 = noise_est()
#     noise_map = net1(x_in)
#     net2_input = concatenate([x_in, noise_map], axis=-1)
#     net2 = ffd_model(False)
#     denoise = net2(net2_input)
# 
#     if not is_train:
#         return Model(x_in, [denoise, noise_map])
#     else :
#         lambd_in = Input(shape=(1,))
#         def custom_loss_1(lambd=1):
#             def tvloss(y):
#                 diff1 = y[1:, :, :] - y[:-1, :, :]
#                 diff2 = y[:, 1:, :] - y[:, :-1, :]
#                 return K.mean(K.square(diff1)) + K.mean(K.square(diff2))
#             def loss(y_true, y_pred):
#                 return finetune_loss(y_true[0], y_pred[0]) + 0.5 * finetune_loss(y_true[1], y_pred[1]) + lambd * tvloss(y_pred[1])
#             return loss
#         def finetune_loss(y_true, y_pred):
#             return K.mean(K.square(y_pred - y_true))
# 
#         model = Model([x_in, lambd_in], [denoise, noise_map])
#         model.compile(loss=custom_loss_1(lambd_in), optimizer='adam')
#         return model

if __name__ == '__main__':
    model = cbd_model()
    model.save_weights('./models/cbd/test.hdf5')

    
