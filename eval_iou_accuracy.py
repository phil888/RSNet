import numpy as np
import glob
import os


def load_obj(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = np.zeros((len(lines), 4))
    for idx in range(len(lines)):
        line = lines[idx]
        line = line.split()[1:]
        line = [float(i) for i in line]
        data[idx, 0], data[idx, 1], data[idx, 2], data[idx, 3] = line[0], line[1], line[2], line[-1]
    return data


def eval(gt_data_label, pred_data_lable):
    num_room = len(gt_data_label)

    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    for i in range(num_room):
        print(i)
        pred_label = pred_data_lable[i]
        gt_label = gt_data_label[i]
        assert len(pred_label) == len(gt_label)
        print(len(gt_label))
        for j in range(len(gt_label)):
            gt_l = int(gt_label[j])
            pred_l = int(pred_label[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)

    oa = sum(true_positive_classes) / float(sum(positive_classes))
    print('Overall accuracy: {0}'.format(sum(true_positive_classes) / float(sum(positive_classes))))
    meanAcc = 0
    for tp, gt in zip(true_positive_classes, gt_classes):
        if gt == 0:
            meanAcc += 0
        else:
            meanAcc += (tp / float(gt))

    meanAcc /= 13
    print('Mean accuracy: {0}'.format(meanAcc))

    print('IoU:')
    iou_list = []
    for i in range(13):
        if float(gt_classes[i] + positive_classes[i] - true_positive_classes[i]) == 0.00:
            if true_positive_classes[i] == 0:
                continue
            else:
                iou = 0
        else:
            iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        print(i, iou)
        iou_list.append(iou)

    print(sum(iou_list) / 13.0)
    meanIOU = sum(iou_list) / 13.0

    with open('test_log.txt', 'a') as f:
        f.write(' OA {:.5f} MA {:.5f} MIOU {:.5f} \n'.format(oa, meanAcc, meanIOU))
