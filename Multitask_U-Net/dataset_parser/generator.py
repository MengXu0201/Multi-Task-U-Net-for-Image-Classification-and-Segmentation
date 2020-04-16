import numpy as np
import random
import cv2
import os
from PIL import Image
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

def pre_processing(img):

    return img / 127.5 - 1
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)
        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)


    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()
    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args


def get_data_gen_args_cls(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args

def get_data_gen_args_seg(mode):
    if mode == 'train' or mode == 'val':
        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        y_data_gen_args = dict()
    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return y_data_gen_args

def data_generator_pair(nameList, segLabel, cLabel, image_shape, nb_class, b_size, mode):
    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)
    First = True
    d_size = len(nameList)
    shuffled_idx = list(range(d_size))
    y_seg = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]
            name = nameList[idx]
            L = int(cLabel[idx])
            img = Image.open(name)
            y_img = load_data(segLabel[idx], image_shape, mode="label")
            img_rgb = img.convert('RGB')
            img_rgb = img_rgb.resize((128, 128), Image.ANTIALIAS)
            img_rgb_arr = np.array(img_rgb)
            img_rgb_arr = np.reshape(img_rgb_arr, [1, 128, 128, 3])
            y = np.zeros([1, nb_class])
            y[0, L] = 1
            if First:
                X_train = img_rgb_arr
                y_train_one = y
                First = False
            else:
                X_train = np.concatenate((X_train, img_rgb_arr), axis=0)
                y_train_one = np.concatenate((y_train_one, y), axis=0)
            y_seg.append(y_img)
            if X_train.shape[0] == b_size:
                y_try = np.argmax(y_train_one, axis=1)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(X_train, y_try,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y_seg), y_try,
                                            batch_size=b_size,
                                            seed=seed)
                x_result, y_try1 = next(x_tmp_gen)
                y_result, y_try2 = next(y_tmp_gen)
                y_train_one1 = np.zeros([y_train_one.shape[0], y_train_one.shape[1]])
                for j in range(b_size):
                    y_train_one1[j, y_try1[j]] = 1.0

                # augmented
                yield [x_result, x_result], [y_train_one1, binarylab(b_size, y_result, image_shape, nb_class)]
                # original
                # yield X_train, y_train_one
                First = True
                y_seg.clear()

def data_generator_pair_different_dataset(cls_nameList, seg_nameList, seg_gt, cLabel, image_shape, nb_class, b_size, mode):
    # Make ImageDataGenerator.

    cls_train_generator = data_generator_cls(nameList=cls_nameList, cLabel=cLabel, image_shape=image_shape, nb_class=nb_class, b_size=1, mode=mode)
    seg_data_gen = data_generator_seg(nameList=seg_nameList, segLabel=seg_gt, img_shape=image_shape, nb_class=nb_class, b_size=1, mode=mode)
    while True:
        cls_imgs = []
        cls_labels = []
        for k in range(b_size):
            cls_data = cls_train_generator.__next__()
            cls_imgs.append(cls_data[0][0])
            cls_labels.append(cls_data[1][0])
        cls_imgs = np.array(cls_imgs)
        cls_labels = np.array(cls_labels)
        # print('cls_img shape:{}'.format(cls_imgs.shape))
        # print('cls_img label shape:{}'.format(cls_labels.shape))
        seg_datas = []
        seg_labels = []
        for k in range(b_size):
            seg_data_lb = seg_data_gen.__next__()
            seg_datas.append(seg_data_lb[0][0])
            seg_labels.append(seg_data_lb[1][0])
        seg_datas = np.array(seg_datas)
        seg_labels = np.array(seg_labels)
        yield [cls_imgs, seg_datas], [cls_labels, seg_labels]

def data_generator_cls(nameList, cLabel, image_shape, nb_class, b_size, mode):
    # Make ImageDataGenerator.
    x_data_gen_args = get_data_gen_args_cls(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    First = True
    d_size = len(nameList)
    shuffled_idx = list(range(d_size))
    y_seg = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]
            name = nameList[idx]
            L = int(cLabel[idx])
            img = Image.open(name)
            img_rgb = img.convert('RGB')
            img_rgb = img_rgb.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
            img_rgb_arr = np.array(img_rgb)
            img_rgb_arr = np.reshape(img_rgb_arr, [1, image_shape[1], image_shape[0], 3])
            y = np.zeros([1, nb_class])
            y[0, L] = 1
            if First:
                X_train = img_rgb_arr
                y_train_one = y
                First = False
            else:
                X_train = np.concatenate((X_train, img_rgb_arr), axis=0)
                y_train_one = np.concatenate((y_train_one, y), axis=0)
            if X_train.shape[0] == b_size:
                y_try = np.argmax(y_train_one, axis=1)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(X_train, y_try,
                                            batch_size=b_size,
                                            seed=seed)

                x_result, y_try1 = next(x_tmp_gen)
                y_train_one1 = np.zeros([y_train_one.shape[0], y_train_one.shape[1]])
                for j in range(b_size):
                    y_train_one1[j, y_try1[j]] = 1.0

                # augmented
                yield x_result, y_train_one1
                # original
                # yield X_train, y_train_one
                First = True
                y_seg.clear()

def data_generator_seg(nameList, segLabel, img_shape, nb_class, b_size, mode):

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = len(nameList)
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]
            x_img = load_data(nameList[idx], img_shape, mode="data")
            y_img = load_data(segLabel[idx], img_shape, mode="label")

            x.append(x_img)
            y.append(y_img)

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, binarylab(b_size, y_result, img_shape, nb_class)

                x.clear()
                y.clear()


def load_data(path, img_shape, mode=None):
    img = Image.open(path)
    img = img.resize((img_shape[1],img_shape[0]))
    #w,h = img.size
    #if w < h:
    #    if w < size:
    #        img = img.resize((size, size*h//w))
    #        w, h = img.size
    #else:
    #    if h < size:
    #        img = img.resize((size*w//h, size))
    #        w, h = img.size
    #img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))
    #img.show()
    if mode=="original":
        return img

    if mode=="label":
        y = np.array(img, dtype=np.int32)
        mask = y == 255
        y[mask] = 21
        #y = binarylab(y, size, 21)
        y = np.expand_dims(y, axis=-1)
        return y
    if mode=="data":
        img = img.convert('RGB')
        X = image.img_to_array(img)
        #X = np.expand_dims(X, axis=0)
        #X = preprocess_input(X)
        return X

def binarylab(b_size, y_img, img_shape, nb_class):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, img_shape[0], img_shape[1], nb_class))

    # For np.where calculation.
    for i in range(nb_class):
        mask = (y_img == i)
        result_map[:, :, :, i] = np.where(mask, 1, 0)

    return result_map