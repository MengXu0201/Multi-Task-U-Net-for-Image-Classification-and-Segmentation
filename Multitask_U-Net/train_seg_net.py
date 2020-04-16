import numpy as np
from model.multiTask import create_pair_model
from model.unet import create_seg_model
from dataset_parser.generator import data_generator_pair, data_generator_seg
from keras.callbacks import ModelCheckpoint

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
    train_gt.extend(train_good_gt)
    train_gt.extend(train_bad_gt)
    clabel_train = np.concatenate([L1_train, L0_train])
    val_name.extend(val_good)
    val_name.extend(val_bad)
    val_gt.extend(val_good_gt)
    val_gt.extend(val_bad_gt)
    clabel_val = np.concatenate([L1_val, L0_val])
    a=1
    train_gen = data_generator_seg(nameList= train_name, segLabel=train_gt,
                                    img_shape=(seg_img_height, seg_img_width, 1), nb_class=2,
                                    b_size=batch_size, mode='train')
    test_gen = data_generator_seg(nameList=val_name, segLabel=val_gt,
                                    img_shape=(seg_img_height, seg_img_width, 1), nb_class=2,
                                    b_size=batch_size, mode='val')
    #
    # train_gen = pair_generator(batch_size=batch_size, cls_img_height=cls_img_height, cls_img_width=cls_img_width, seg_train_dir=seg_path_to_train_img,
    #                            seg_label_dir=seg_path_to_train_label, cls_samples_dir=cls_train_data_dir, seg_train_names=seg_namestrain)
    # test_gen = pair_generator(batch_size=batch_size, cls_img_height=cls_img_height, cls_img_width=cls_img_width,
    #                            seg_train_dir=seg_path_to_train_img,
    #                            seg_label_dir=seg_path_to_train_label, cls_samples_dir=cls_validation_data_dir,
    #                            seg_train_names=seg_namestest)
    pair_model = create_seg_model(seg_width=seg_img_width,
                                   seg_height=seg_img_height)
    checkpoint = ModelCheckpoint(filepath='./log/test_model_checkpoint_weight.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True)

    pair_model.fit_generator(train_gen,
                            steps_per_epoch=500 // batch_size,
                            epochs=epochs,
                            validation_data=test_gen,
                            validation_steps=500 // batch_size,
                            callbacks=[checkpoint])
    # # plot_model(model=pair_model, to_file='pair-model', show_shapes=True)
    # # exit(0)
    # pair_model = train_pair_model(model=pair_model, train_generator=train_gen,
    #                               val_generator=test_gen, epochs=epochs, batch_size=batch_size)
    # pair_model.save_weights(pair_weight_path)
    # # pair_model.save_weights('./pair2_model_weights-{}.h5'.format(data_select))
    # # with open('./pair2_-255model_struct-255.json', 'w') as struct_file:
    # #     struct_file.write(pair_model.to_json())
    # with open(pair_struct_path, 'w') as struct_file:
    #     struct_file.write(pair_model.to_json())
    #
    #
    # evaluate.eval_pair2(cls_img_width=cls_img_width, cls_img_height=cls_img_height,
    #                     seg_img_width=seg_img_width, seg_img_height=seg_img_height,
    #                     pair_struct_path=pair_struct_path, pair_weight_path=pair_weight_path,
    #                     pdt_save_dir=pdt_save_dir,
    #                     seg_test_dir=seg_test_dir,
    #                     seg_label_dir=seg_label_dir,
    #                     seg_test_name_path=seg_test_name_path)
    # # seg_model = load_seg_model(pair_model, seg_img_width, seg_img_height)
    # # cls_model = load_cls_model(pair_model, height=cls_img_height, width=cls_img_width)

# def train_pair_model(model, train_generator, val_generator, epochs, batch_size, log_dir='pair2_-255_logs'):
#     # checkpoint = ModelCheckpoint(model_cache_name,
#     #                              monitor='val_acc',
#     #                              verbose=1,
#     #                              save_best_only=True,
#     #                              mode='max')
#     log_dir = os.path.join(rootPath, log_dir)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     callback_tb = TensorBoard(log_dir=log_dir)
#     cls_callback = my_callbacks.MultiTaskCLSCallback(val_acc_threshold=0.7, cls_good_rate_threshold=0.75, cls_bad_rate_threadhold=0.6, model_cache_dir=log_dir, eval=True,
#                                                      eval_good_dir='../cls-samples/bound-new/test-image/good', eval_bad_dir='../cls-samples/bound-new/test-image/bad')
#     callbacks_list = [cls_callback, callback_tb]
#     model.fit_generator(train_generator,
#                         steps_per_epoch= 500 // batch_size,
#                         epochs=epochs,
#                         validation_data=val_generator,
#                         validation_steps=500 // batch_size,
#                         callbacks=callbacks_list)
#     return model
if __name__ == '__main__':
    # eval_test()
    train_seg_model_handler()
    # eval_cls_from_path('./cls_model_pair.hdf5', 243, 243, None)