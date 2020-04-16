from keras.layers import Input, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, Activation, concatenate, Dense, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from model.unet import dice_coef
import keras.backend as K
from keras.models import Model

def binary_crossentropy(y_true, y_pred):
    e=1.0
    return K.mean(-(y_true*K.log(y_pred+K.epsilon())+
                    e*(1-y_true)*K.log(1-y_pred+K.epsilon())),
                  axis=-1)

def create_cls_model(cls_width, cls_height):
    cls_input = Input(shape=(cls_height, cls_width, 3), name='cls_input')

    conv1 = Conv2D(64, (3, 3), padding='same', name='share_conv1_1')
    conv1_cls = conv1(cls_input)

    x = Activation('relu', name='cls_act_0')(conv1_cls)


    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_0')(x)

    x = Conv2D(64, (3, 3), name='cls_conv_1')(x)
    x = Activation('relu', name='cls_actv_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_1')(x)

    x = Conv2D(128, (3, 3), name='cls_conv_2')(x)
    x = Activation('relu', name='cls_act_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_2')(x)

    x = Flatten(name='cls_flatten')(x)
    x = Dense(64, name='cls_dense_0')(x)
    x = Activation('relu', name='cls_act_3')(x)
    x = Dropout(0.5, name='cls_dropout')(x)
    x = Dense(2, name='cls_dense_out')(x)
    out_cls = Activation('sigmoid', name='cls_out')(x)

    model = Model(inputs=cls_input, outputs=out_cls)

    # layer = Layer
    # idx = 0
    # for layer in model.layers:
    #     print('{}:{}'.format(idx, layer.name))
    #     idx = idx + 1
    # plot_model(model, to_file='pair-model_2.png', show_shapes=True)

    model.summary()
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model