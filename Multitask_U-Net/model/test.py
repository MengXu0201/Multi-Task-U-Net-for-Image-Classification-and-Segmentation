from keras.layers import Input, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, Activation, concatenate, Dense, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from model.unet import dice_coef
import keras.backend as K
from keras.models import Model


def binary_crossentropy(y_true, y_pred):
    e = 1.0
    return K.mean(-(y_true * K.log(y_pred + K.epsilon()) +
                    e * (1 - y_true) * K.log(1 - y_pred + K.epsilon())),
                  axis=-1)


def create_pair_model(seg_width, seg_height, cls_width, cls_height, seg_loss_weight=0.5, cls_loss_weight=0.5):
    cls_input = Input(shape=(cls_height, cls_width, 3), name='cls_input')
    seg_input = Input(shape=(seg_height, seg_width, 3), name='seg_input')
    # assume cls_input is the same as seg_input
    # shared layers
    # block 1
    shared1 = Conv2D(64, (3, 3), padding='same', name='share_conv1_1')
    shared2 = BatchNormalization()
    shared3 = Activation('relu')

    x_1 = shared1(cls_input)
    x_1 = shared2(x_1)
    x_1 = shared3(x_1)

    shared = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(shared)
    shared = BatchNormalization()(shared)

    # unet block 1 output
    block_1_out = Activation('relu')(shared)
    shared = MaxPooling2D()(block_1_out)

    # block 2
    shared = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(shared)
    shared = BatchNormalization()(shared)
    block_2_out = Activation('relu')(shared)

    shared = MaxPooling2D()(block_2_out)

    # block 3
    shared = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(shared)
    shared = BatchNormalization()(shared)
    block_3_out = Activation('relu')(shared)

    shared = MaxPooling2D()(block_3_out)

    # block 4
    shared = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(shared)
    shared = BatchNormalization()(shared)
    block_4_out = Activation('relu')(shared)

    shared = MaxPooling2D()(block_4_out)

    # Block 5
    shared = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    shared = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(shared)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)

    # cls net
    conv1_cls = shared

    x = Flatten(name='cls_flatten')(conv1_cls)
    x = Dense(64, name='cls_dense_0')(x)
    x = Activation('relu', name='cls_act_3')(x)
    x = Dropout(0.5, name='cls_dropout')(x)
    x = Dense(2, name='cls_dense_out')(x)
    out_cls = Activation('sigmoid', name='cls_out')(x)

    # seg net
    conv1_seg = shared

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv1_seg)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    out_seg = Conv2D(2, (3, 3), activation='softmax', padding='same', name='seg_out')(x)
    # out_seg = Activation(activation='softmax', name='seg-out')(conv10)

    model = Model(inputs=[cls_input, seg_input], outputs=[out_cls, out_seg])

    # layer = Layer
    # idx = 0
    # for layer in model.layers:
    #     print('{}:{}'.format(idx, layer.name))
    #     idx = idx + 1
    # plot_model(model, to_file='pair-model_2.png', show_shapes=True)

    model.summary()
    model.compile(loss={'seg_out': 'categorical_crossentropy', 'cls_out': binary_crossentropy},
                  optimizer=Adam(lr=1e-4), metrics={'seg_out': dice_coef, 'cls_out': 'accuracy'},
                  loss_weights={'seg_out': seg_loss_weight, 'cls_out': cls_loss_weight})
    return model