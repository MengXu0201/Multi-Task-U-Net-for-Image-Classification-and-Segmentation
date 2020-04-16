import numpy as np
from model.multiTask import create_pair_model
from model.unet import create_seg_model
from dataset_parser.generator import data_generator_pair, data_generator_seg
from keras.callbacks import ModelCheckpoint
from model.unetkuan import unet
def train_seg_model_handler():

    model_name = 'segmentation net'
    seg_loss_weight = 0.5
    print('seg loss:{}'.format(seg_loss_weight))
    loss_str = 'loss_seg_{}'.format(str(seg_loss_weight).replace('.', '_'))
    # data_select = '325' #0 :255 1:325
    seg_img_width = 256
    seg_img_height = 256
    batch_size = 11
    epochs = 80
    data1 = '../BUS/data1/'
    data2 = '../BUS/data2/'
    train_data_dir_data1_good = data1 + 'data1_good_train.txt'
    validation_data_dir_data1_good = data1 + 'data1_good_val.txt'
    train_data_dir_data1_bad = data1 + 'data1_bad_train.txt'
    validation_data_dir_data1_bad = data1 + 'data1_bad_val.txt'
    train_data_dir_data2_good = data2 + 'data2_good_train.txt'
    validation_data_dir_data2_good = data2 + 'data2_good_val.txt'
    train_data_dir_data2_bad = data2 + 'data2_bad_train.txt'
    validation_data_dir_data2_bad = data2 + 'data2_bad_val.txt'

    # Add dataset3
    seg_data = '../BUS/data3/'
    seg_train = seg_data + 'train.txt'
    seg_test = seg_data + 'test.txt'

# data1
    with open(train_data_dir_data1_good, "r") as f:
        ls = f.readlines()
    data1_good_train = [l.rstrip('\n') for l in ls]
    data1_good_train_original = []
    data1_good_train_seggt = []
    for i in range(len(data1_good_train)):
        data1_good_train_original.append('../BUS/data1/original/' + data1_good_train[i] + '.png')
        data1_good_train_seggt.append('../BUS/data1/GT/' + data1_good_train[i] + '.png')
    with open(validation_data_dir_data1_good, "r") as f:
        ls = f.readlines()
    data1_good_val = [l.rstrip('\n') for l in ls]
    data1_good_val_original = []
    data1_good_val_seggt = []
    for i in range(len(data1_good_val)):
        data1_good_val_original.append('../BUS/data1/original/' + data1_good_val[i] + '.png')
        data1_good_val_seggt.append('../BUS/data1/GT/' + data1_good_val[i] + '.png')

    with open(train_data_dir_data1_bad, "r") as f:
        ls = f.readlines()
    data1_bad_train = [l.rstrip('\n') for l in ls]
    data1_bad_train_original = []
    data1_bad_train_seggt = []
    for i in range(len(data1_bad_train)):
        data1_bad_train_original.append('../BUS/data1/original/' + data1_bad_train[i] + '.png')
        data1_bad_train_seggt.append('../BUS/data1/GT/' + data1_bad_train[i] + '.png')
    with open(validation_data_dir_data1_bad, "r") as f:
        ls = f.readlines()
    data1_bad_val = [l.rstrip('\n') for l in ls]
    data1_bad_val_original = []
    data1_bad_val_seggt = []
    for i in range(len(data1_bad_val)):
        data1_bad_val_original.append('../BUS/data1/original/' + data1_bad_val[i] + '.png')
        data1_bad_val_seggt.append('../BUS/data1/GT/' + data1_bad_val[i] + '.png')
##data2
    with open(train_data_dir_data2_good, "r") as f:
        ls = f.readlines()
    data2_good_train = [l.rstrip('\n') for l in ls]
    data2_good_train_original = []
    data2_good_train_seggt = []
    for i in range(len(data2_good_train)):
        data2_good_train_original.append('../BUS/data2/original/C' + data2_good_train[i][1:] + '.png')
        data2_good_train_seggt.append('../BUS/data2/GT/C' + data2_good_train[i][1:] + '.png')
    with open(validation_data_dir_data2_good, "r") as f:
        ls = f.readlines()
    data2_good_val = [l.rstrip('\n') for l in ls]
    data2_good_val_original = []
    data2_good_val_seggt = []
    for i in range(len(data2_good_val)):
        data2_good_val_original.append('../BUS/data2/original/C' + data2_good_val[i][1:] + '.png')
        data2_good_val_seggt.append('../BUS/data2/GT/C' + data2_good_val[i][1:] + '.png')

    with open(train_data_dir_data2_bad, "r") as f:
        ls = f.readlines()
    data2_bad_train = [l.rstrip('\n') for l in ls]
    data2_bad_train_original = []
    data2_bad_train_seggt = []
    for i in range(len(data2_bad_train)):
        data2_bad_train_original.append('../BUS/data2/original/C' + data2_bad_train[i][1:] + '.png')
        data2_bad_train_seggt.append('../BUS/data2/GT/C' + data2_bad_train[i][1:] + '.png')
    with open(validation_data_dir_data2_bad, "r") as f:
        ls = f.readlines()
    data2_bad_val = [l.rstrip('\n') for l in ls]
    data2_bad_val_original = []
    data2_bad_val_seggt = []
    for i in range(len(data2_bad_val)):
        data2_bad_val_original.append('../BUS/data2/original/C' + data2_bad_val[i][1:] + '.png')
        data2_bad_val_seggt.append('../BUS/data2/GT/C' + data2_bad_val[i][1:]+ '.png')

    # Add dataset 3
    # Segment dataset
    with open(seg_train, "r") as f:
        ls = f.readlines()
    seg_train_name = [l.rstrip('\n') for l in ls]
    with open(seg_test, "r") as f:
        ls = f.readlines()
    seg_test_name = [l.rstrip('\n') for l in ls]
    seg_train_add = []
    seg_train_gt_add = []
    for i in range(len(seg_train_name)):
        seg_train_add.append('../BUS/data3/original/c' + seg_train_name[i][1:] + '.png')
        seg_train_gt_add.append('../BUS/data3/GT/c' + seg_train_name[i][1:] + '_GT.png')
    seg_test_add = []
    seg_test_gt_add = []
    for i in range(len(seg_test_name)):
        seg_test_add.append('../BUS/data3/original/c' + seg_test_name[i][1:] + '.png')
        seg_test_gt_add.append('../BUS/data3/GT/c' + seg_test_name[i][1:] + '_GT.png')
    a = 1

    train_good = []
    train_good.extend(data1_good_train_original)
    train_good.extend(data2_good_train_original)
    train_bad = []
    train_bad.extend(data1_bad_train_original)
    train_bad.extend(data2_bad_train_original)

    val_good = []
    val_bad = []
    val_good.extend(data1_good_val_original)
    val_good.extend(data2_good_val_original)
    val_bad.extend(data1_bad_val_original)
    val_bad.extend(data2_bad_val_original)
    train_good_gt = []
    train_bad_gt = []
    val_good_gt = []
    val_bad_gt = []
    train_good_gt.extend(data1_good_train_seggt)
    train_good_gt.extend(data2_good_train_seggt)
    train_bad_gt.extend(data1_bad_train_seggt)
    train_bad_gt.extend(data2_bad_train_seggt)
    val_good_gt.extend(data1_good_val_seggt)
    val_good_gt.extend(data2_good_val_seggt)
    val_bad_gt.extend(data1_bad_val_seggt)
    val_bad_gt.extend(data2_bad_val_seggt)
    L1_train = np.ones(len(train_good_gt))
    L1_val = np.ones(len(val_good_gt))
    L0_train = np.zeros(len(train_bad_gt))
    L0_val = np.zeros(len(val_bad_gt))
    train_name = []
    train_gt = []
    val_name = []
    val_gt = []
    train_name.extend(train_good)
    train_name.extend(train_bad)
    # added dataset3
    train_name.extend(seg_train_add)
    train_gt.extend(train_good_gt)
    train_gt.extend(train_bad_gt)
    # added dataset3
    train_gt.extend(seg_train_gt_add)
    clabel_train = np.concatenate([L1_train, L0_train])
    val_name.extend(val_good)
    val_name.extend(val_bad)
    # added dataset3
    val_name.extend(seg_test_add)
    val_gt.extend(val_good_gt)
    val_gt.extend(val_bad_gt)
    # added dataset3
    val_gt.extend(seg_test_gt_add)
    clabel_val = np.concatenate([L1_val, L0_val])
    a=1
    model = unet(input_shape=(seg_img_height, seg_img_width, 3), num_classes=2,
                 lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=None)
    checkpoint = ModelCheckpoint(filepath='./log/unet_model_checkpoint_weight.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True)
    history = model.fit_generator(
        data_generator_seg(nameList=train_name, segLabel=train_gt,
                           img_shape=(seg_img_height, seg_img_width, 3), nb_class=2,
                           b_size=batch_size, mode='train'),
        steps_per_epoch=len(train_name) // batch_size,
        validation_data=data_generator_seg(nameList=val_name, segLabel=val_gt,
                                    img_shape=(seg_img_height, seg_img_width, 3), nb_class=2,
                                    b_size=batch_size, mode='val'),
        validation_steps=len(val_name) // 1,
        callbacks=[checkpoint],
        epochs=epochs,
        verbose=1)
    model.save_weights('./log/unet_model_weight.h5')
if __name__ == '__main__':
    # eval_test()
    train_seg_model_handler()
    # eval_cls_from_path('./cls_model_pair.hdf5', 243, 243, None)