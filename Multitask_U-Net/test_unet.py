import numpy as np
from PIL import Image
from model.unet import create_seg_model
from model.unetkuan import unet
from dataset_parser.prepareData import VOCPalette
import matplotlib.pyplot as plt
import cv2
def seg_result(res_map):
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2).astype('uint8')
    return argmax_idx

def load_mask(path,img_shape):
    img = Image.open(path)
    #img = img.resize((img_shape[1], img_shape[0]))
    y = np.array(img, dtype=np.int32)
    mask = y == 255
    y[mask] = 21
    y = np.expand_dims(y, axis=-1)
    return y

# def seg_accuracy(seg_predict, seg_gt_path, img_shape):
#     mask1 = seg_predict
#     mask2 = Image.open(seg_gt_path)
#     mask2 = np.array(mask2, dtype=np.int32)
#     #mask2 = mask2[:,:,0]

def seg_accuracy(seg_predict, seg_gt_path, img_shape):
    mask1 = seg_predict
    mask2 = load_mask(seg_gt_path, img_shape)
    mask2 = mask2[:, :, 0]

    n1 = mask1 == 1
    n2 = mask2 == 1
    intersection = n1 & n2
    union = n1 | n2
    intersection = np.uint8(intersection)
    union = np.uint8(union)
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    overlaps = intersection_sum / union_sum
    # intersection = [val for val in mask1 if val in mask2]
    # union = np.array(list(set(mask1).union(set(mask2))))
    #
    # overlaps = intersection / union
    return overlaps


def test_seg_model_handler():
    class_num = ['Malignant', 'Benign']
    seg_img_width = 256
    seg_img_height = 256
    seg_loss_weight = 0.5
    cls_loss_weight = 0.5
    nb_class = 2
    image_shape = (seg_img_height, seg_img_width, 3)

    # load saved model
    model_name = './log/unet_model_weight.h5'
    pair_model = unet(input_shape=(seg_img_height, seg_img_width, 3), num_classes=2,
                 lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=None)
    try:
        pair_model.load_weights(model_name)
    except:
        print("You must train model and get weight before test.")

    # # paths to validation set
    # img_path = '../BUS/data2/original/Case27.png'
    # labe_path = '../BUS/data2/GT/Case27.png'
    #
    # palette = VOCPalette(nb_class=nb_class)
    # imgorg = Image.open(img_path)
    # imglab = Image.open(labe_path)
    # imgorg = imgorg.convert('RGB')
    # img_cls = imgorg.resize((cls_img_width, cls_img_height), Image.ANTIALIAS)
    # img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)
    # img_cls_arr = np.array(img_cls)
    # img_cls_arr = img_cls_arr / 127.5 - 1
    # img_seg_arr = np.array(img_seg)
    # img_seg_arr = img_seg_arr / 127.5 - 1
    # img_cls_arr = np.expand_dims(img_cls_arr, 0)
    # img_seg_arr = np.expand_dims(img_seg_arr, 0)
    # # predict results
    # pred = pair_model.predict([img_cls_arr, img_seg_arr])
    # cls_r = cls_result(pred[0])
    # seg_r = seg_result(pred[1])
    # # plot the predicted results.
    # PIL_img_pal = palette.genlabelpal(seg_r)
    # PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)
    # plt.ion()
    # plt.figure('Multi task')
    # plt.suptitle(img_path + '\n' + 'Class:' + class_num[cls_r])
    # plt.subplot(1, 3, 1), plt.title('org')
    # plt.imshow(imgorg), plt.axis('off')
    # plt.subplot(1, 3, 2), plt.title('segmentation result')
    # plt.imshow(PIL_img_pal), plt.axis('off')
    # plt.subplot(1, 3, 3), plt.title('label')
    # plt.imshow(imglab), plt.axis('off')
    # plt.show()

    # collect validation data
    data1 = '../BUS/data1/'
    data2 = '../BUS/data2/'
    validation_data_dir_data1_good = data1 + 'data1_good_val.txt'
    validation_data_dir_data1_bad = data1 + 'data1_bad_val.txt'
    validation_data_dir_data2_good = data2 + 'data2_good_val.txt'
    validation_data_dir_data2_bad = data2 + 'data2_bad_val.txt'

    # Add dataset3
    seg_data = '../BUS/data3/'
    seg_test = seg_data + 'test.txt'

    # Add dataset 3
    # Segment dataset
    with open(seg_test, "r") as f:
        ls = f.readlines()
    seg_test_name = [l.rstrip('\n') for l in ls]

    seg_test_add = []
    seg_test_gt_add = []
    for i in range(len(seg_test_name)):
        seg_test_add.append('../BUS/data3/original/c' + seg_test_name[i][1:] + '.png')
        seg_test_gt_add.append('../BUS/data3/GT/c' + seg_test_name[i][1:] + '_GT.png')
    a = 1

    # data1 good
    with open(validation_data_dir_data1_good, "r") as f:
        ls = f.readlines()
    data1_good_val = [l.rstrip('\n') for l in ls]
    data1_good_val_original = []
    data1_good_val_seggt = []
    for i in range(len(data1_good_val)):
        data1_good_val_original.append('../BUS/data1/original/' + data1_good_val[i] + '.png')
        data1_good_val_seggt.append('../BUS/data1/GT/' + data1_good_val[i] + '.png')
    # data1 bad
    with open(validation_data_dir_data1_bad, "r") as f:
        ls = f.readlines()
    data1_bad_val = [l.rstrip('\n') for l in ls]
    data1_bad_val_original = []
    data1_bad_val_seggt = []
    for i in range(len(data1_bad_val)):
        data1_bad_val_original.append('../BUS/data1/original/' + data1_bad_val[i] + '.png')
        data1_bad_val_seggt.append('../BUS/data1/GT/' + data1_bad_val[i] + '.png')

    # data2 good
    with open(validation_data_dir_data2_good, "r") as f:
        ls = f.readlines()
    data2_good_val = [l.rstrip('\n') for l in ls]
    data2_good_val_original = []
    data2_good_val_seggt = []
    for i in range(len(data2_good_val)):
        data2_good_val_original.append('../BUS/data2/original/C' + data2_good_val[i][1:] + '.png')
        data2_good_val_seggt.append('../BUS/data2/GT/C' + data2_good_val[i][1:] + '.png')

    # data2 bad
    with open(validation_data_dir_data2_bad, "r") as f:
        ls = f.readlines()
    data2_bad_val = [l.rstrip('\n') for l in ls]
    data2_bad_val_original = []
    data2_bad_val_seggt = []
    for i in range(len(data2_bad_val)):
        data2_bad_val_original.append('../BUS/data2/original/C' + data2_bad_val[i][1:] + '.png')
        data2_bad_val_seggt.append('../BUS/data2/GT/C' + data2_bad_val[i][1:]+ '.png')

    val_good = []
    val_bad = []
    val_good.extend(data1_good_val_original)
    val_good.extend(data2_good_val_original)
    val_bad.extend(data1_bad_val_original)
    val_bad.extend(data2_bad_val_original)
    val_good_gt = []
    val_bad_gt = []
    val_good_gt.extend(data1_good_val_seggt)
    val_good_gt.extend(data2_good_val_seggt)
    val_bad_gt.extend(data1_bad_val_seggt)
    val_bad_gt.extend(data2_bad_val_seggt)

    val_name = []
    val_gt = []
    # path and file name of images in the validation set
    val_name.extend(val_good)
    val_name.extend(val_bad)
    val_name.extend(seg_test_add)
    # segmentation gt of validation set
    val_gt.extend(val_good_gt)
    val_gt.extend(val_bad_gt)
    val_gt.extend(seg_test_gt_add)
    # number of good and bad images
    L1_val = np.ones(len(val_good_gt))
    L0_val = np.zeros(len(val_bad_gt))
    # number of images in validation set
    num_val = len(val_good_gt) + len(val_bad_gt)

    # compute accuracy of the validation set
    good_good = 0
    good_bad = 0
    bad_good = 0
    bad_bad = 0
    overlaps = []

    # data1 good
    for index, name in enumerate(data1_good_val_original):
        # path and file name of current image
        img_path = data1_good_val_original[index]
        # segmentation ground truth
        label_path = data1_good_val_seggt[index]

        palette = VOCPalette(nb_class=nb_class)
        imgorg = Image.open(img_path)
        imglab = Image.open(label_path)
        #imgorg = imgorg.convert('RGB')

        img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)
        img_seg_arr = np.array(img_seg)
        img_seg_arr = img_seg_arr / 127.5 - 1
        img_seg_arr = np.expand_dims(img_seg_arr, 0)
        #img_seg_arr = np.expand_dims(img_seg_arr, -1)
        # predict results
        pred = pair_model.predict(img_seg_arr)
        seg_r = seg_result(pred)
        seg_res = cv2.resize(seg_r, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
        # segmentation accuracy
        iou = seg_accuracy(seg_res, label_path, image_shape)
        overlaps.append(iou)

        # # plot the predicted results.
        # PIL_img_pal = palette.genlabelpal(seg_r)
        # PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)
        # plt.ion()
        # plt.figure('Multi task')
        # # plt.suptitle(img_path + '\n' + 'Class:' + class_num[cls_r])
        # plt.subplot(1, 3, 1), plt.title('org')
        # plt.imshow(imgorg), plt.axis('off')
        # plt.subplot(1, 3, 2), plt.title('segmentation result')
        # plt.imshow(PIL_img_pal), plt.axis('off')
        # plt.subplot(1, 3, 3), plt.title('label')
        # plt.imshow(imglab), plt.axis('off')
        # plt.show()
    a = 1

    # data1 bad
    for index, name in enumerate(data1_bad_val_original):
        # path and file name of current image
        img_path = data1_bad_val_original[index]
        # segmentation ground truth
        label_path = data1_bad_val_seggt[index]

        palette = VOCPalette(nb_class=nb_class)
        imgorg = Image.open(img_path)
        imglab = Image.open(label_path)
        #imgorg = imgorg.convert('RGB')

        img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)

        img_seg_arr = np.array(img_seg)
        img_seg_arr = img_seg_arr / 127.5 - 1
        img_seg_arr = np.expand_dims(img_seg_arr, 0)
        #img_seg_arr = np.expand_dims(img_seg_arr, -1)
        # predict results
        pred = pair_model.predict(img_seg_arr)
        seg_r = seg_result(pred)
        seg_res = cv2.resize(seg_r, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
        # segmentation accuracy
        iou = seg_accuracy(seg_res, label_path, image_shape)
        overlaps.append(iou)
    a = 1

    # data2 good
    for index, name in enumerate(data2_good_val_original):
        # path and file name of current image
        img_path = data2_good_val_original[index]
        # segmentation ground truth
        label_path = data2_good_val_seggt[index]

        palette = VOCPalette(nb_class=nb_class)
        imgorg = Image.open(img_path)
        imglab = Image.open(label_path)
        #imgorg = imgorg.convert('RGB')
        img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)

        img_seg_arr = np.array(img_seg)
        img_seg_arr = img_seg_arr / 127.5 - 1
        img_seg_arr = np.expand_dims(img_seg_arr, 0)
        #img_seg_arr = np.expand_dims(img_seg_arr, -1)
        # predict results
        pred = pair_model.predict( img_seg_arr)
        seg_r = seg_result(pred)
        seg_res = cv2.resize(seg_r, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
        # segmentation accuracy
        iou = seg_accuracy(seg_res, label_path, image_shape)
        overlaps.append(iou)
    a = 1

    # data2 bad
    for index, name in enumerate(data2_bad_val_original):
        # path and file name of current image
        img_path = data2_bad_val_original[index]
        # segmentation ground truth
        label_path = data2_bad_val_seggt[index]

        palette = VOCPalette(nb_class=nb_class)
        imgorg = Image.open(img_path)
        imglab = Image.open(label_path)
        #imgorg = imgorg.convert('RGB')
        img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)

        img_seg_arr = np.array(img_seg)
        img_seg_arr = img_seg_arr / 127.5 - 1

        img_seg_arr = np.expand_dims(img_seg_arr, 0)
        #img_seg_arr = np.expand_dims(img_seg_arr, -1)
        # predict results
        pred = pair_model.predict(img_seg_arr)
        seg_r = seg_result(pred)
        seg_res = cv2.resize(seg_r, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
        # segmentation accuracy
        iou = seg_accuracy(seg_res, label_path, image_shape)
        overlaps.append(iou)
    a = 1

    # added dataset 3
    for index, name in enumerate(seg_test_add):
        # path and file name of current image
        img_path = seg_test_add[index]
        # segmentation ground truth
        label_path = seg_test_gt_add[index]

        palette = VOCPalette(nb_class=nb_class)
        imgorg = Image.open(img_path)
        imglab = Image.open(label_path)
        # imgorg = imgorg.convert('RGB')
        img_seg = imgorg.resize((seg_img_width, seg_img_height), Image.ANTIALIAS)

        img_seg_arr = np.array(img_seg)
        img_seg_arr = img_seg_arr / 127.5 - 1

        img_seg_arr = np.expand_dims(img_seg_arr, 0)
        #img_seg_arr = np.expand_dims(img_seg_arr, -1)
        # predict results
        pred = pair_model.predict(img_seg_arr)
        seg_r = seg_result(pred)
        seg_res = cv2.resize(seg_r, dsize=(imgorg.size[0], imgorg.size[1]), interpolation=cv2.INTER_LINEAR)
        # segmentation accuracy
        iou = seg_accuracy(seg_res, label_path, image_shape)
        overlaps.append(iou)
    a = 1

    # segmentation accuracy
    mOverlaps = np.mean(overlaps)
    print("mOverlaps: ", mOverlaps)
    a = 1

if __name__ == '__main__':
    # eval_test()
    test_seg_model_handler()
    plt.close('all')
