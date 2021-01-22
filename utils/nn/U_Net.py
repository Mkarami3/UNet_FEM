import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.backend as K

class UNet:

    @staticmethod
    def build(input_size, pretrained_weights = None):

        inputs = Input(input_size)

        conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        drop1 = Dropout(0.5)(conv1)
        conv1 = ZeroPadding3D(padding = ( (0, 1), (0, 1), (0, 1)) )(drop1)

        #Downsampling
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

        #Upsampling
        up1 = UpSampling3D(size = (2, 2, 2))(conv2)
        conv3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up1)
        conv3 = Conv3D(filters=64, kernel_size=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        merge1 = concatenate([conv1,conv3], axis = 4)
        merge1 = Cropping3D(cropping = ( (0, 1), (0, 1), (0, 1)) )(merge1)

        conv4 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
        conv5 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        output = Conv3D(filters=3, kernel_size=(1, 1, 1), activation = 'linear')(conv5)

        model = Model(input = inputs, output = output)        
        model.summary()

        
        opt = Adam(lr=1e-6)
        # model.compile(loss="mse", optimizer=opt, metrics=['mse'])
        model.compile(loss=UNet.custom_loss(inputs), optimizer=opt, metrics=['mse'])

        if(pretrained_weights):
        	model.load_weights(pretrained_weights)

        return model

    #input_data is external force, y_pred is displacement
    def custom_loss(input_data):
        def loss(y_true, y_pred):

            mse_error = UNet.mse_loss #Typical Loss function
            physical_error = UNet.physical_loss(input_data, y_pred)

            return mse_error+physical_error # regularization value = 0.001

        return loss

    def mse_loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def physical_loss(force, disp):
        '''
        if vector product with disp is positive, then loss=0
        if vector product with disp is negative, then loss=sigmoid(force*disp)
        '''
        force_dot_disp = K.sum(force * disp, axis = -1)
        is_positive = force_dot_disp > 0

        loss_for_positive = tf.zeros_like(force_dot_disp,dtype='float32')
        loss_for_negative = K.sigmoid(force_dot_disp)

        return tf.where(is_positive, loss_for_positive, loss_for_negative)
        
    # def physical_loss(force, disp):

    #     force_dot_disp = K.sum(force * disp, axis = -1)
    #     force_norm = K.sqrt(K.sum(K.square(force), axis = -1))
    #     disp_norm = K.sqrt(K.sum(K.square(disp), axis = -1)) 
    #     norm = (force_norm * disp_norm)# to ignore nan values
    #     cos_th = force_dot_disp/norm
    #     # cos_th = force_dot_disp
    #     zero_tensor = tf.zeros_like(cos_th,dtype='float32')
    #     mask = tf.not_equal(cos_th, zero_tensor) #find non-zero values
    #     identity_tensor = tf.cast(mask, tf.float32)

    #     loss =  K.subtract(identity_tensor,  cos_th)
    #     return loss

