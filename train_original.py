import argparse
import os
import shutil
import time
import sys
import numpy as np
import argparse
from torchvision.transforms import ColorJitter, Compose, ToPILImage, ToTensor

from eval_iou_accuracy import eval

seed = 42
np.random.seed(seed)

import shutil

parser = argparse.ArgumentParser(description='Process input arguments.')

#-- arguments for dataset
parser.add_argument('--data_dir', default='./data',
                    help='directory to where processed data is stored')

parser.add_argument('--bs', default=1.0,
                    help='size of each block')

parser.add_argument('--stride', default=0.5,
                    help='stride of block')

parser.add_argument('--area', default='Area_5',
                    help='which area to be used as test set, options: Area_1/Area_2/Area_3/Area_4/Area_5/Area_6')

#-- arguments for RSNet setting
parser.add_argument('--rx', default=0.02,
                    help='slice resolution in x axis')

parser.add_argument('--ry', default=0.02,
                    help='slice resolution in y axis')

parser.add_argument('--rz', default=0.02,
                    help='slice resolution in z axis')

#-- arguments for training settings
parser.add_argument('--lr', default=0.001,
                    help='learning rate')

parser.add_argument('--epochs', default=60,
                    help='epochs')

parser.add_argument('--batchsize', default=24,
                    help='epochs')

parser.add_argument('--weight_file', default='',
                    help='weights to load')

#-- other arguments
parser.add_argument('--gpu', default='0',
                    help='gpu index to use')

parser.add_argument('--model_dir', default='./models',
                    help='folder to hold checkpoints')

parser.add_argument('--results_dir', default='./results',
                    help='folder to hold results')

args = parser.parse_args()

args.area = ['Area_5']
area_name = 'Area_5'


#-- set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

#-- import basic libs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable

#-- import RSNet utils
from net import RSNet
from utils import *
import load_data_aug
from load_data_aug import iterate_data, gen_slice_idx, reset

#-- make directories if neccessary
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

#-- helper func for rnn units
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

#-- load testing meta-data
block_size = float(args.bs)
stride = float(args.stride)

root_file_name = os.path.join(args.data_dir , 'indoor3d_sem_seg_hdf5_data_{}_{}m_{}s_test/room_filelist.txt'.format(area_name,block_size, block_size) )

f = open(root_file_name)
con = f.read().split()
f.close()

#-- load visualization colors
g_classes = [x.rstrip() for x in open( './data/utils/meta/class_names.txt')]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'ceiling':    [0,255,0],
                 'floor':    [0,0,255],
                 'wall':    [0,255,255],
                 'beam':        [255,255,0],
                 'column':      [255,0,255],
                 'window':      [100,100,255],
                 'door':        [200,200,100],
                 'table':       [170,120,200],
                 'chair':       [255,0,0],
                 'sofa':        [200,100,100],
                 'bookcase':    [10,200,100],
                 'board':       [200,200,200],
                 'clutter':     [50,50,50]}

g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


#-- set training settings
lr = args.lr
start_epoch = 0
epochs = args.epochs
best_prec1 = 0
batchsize = args.batchsize


#-- specify slice resolution
RANGE_X, RANGE_Y, RANGE_Z = args.bs, args.bs, load_data_aug.Z_MAX
#- true slice resolution
resolution_true = [args.rx, args.ry, args.rz]
#- modified resolution for easy indexing
resolution = [ i + 0.00001 for i in resolution_true ]
num_slice = [0,0,0]
num_slice[0] = int( RANGE_X / resolution[0] ) + 1
num_slice[1] = int( RANGE_Y / resolution[1] ) + 1
num_slice[2] = int( RANGE_Z / resolution[2] ) + 1

pool_type = 'Max_Pool'

model = RSNet(pool_type, num_slice)
model = model.cuda()

#- disable cudnn. cudnn raises error here due to irregular number of slices
cudnn.benchmark = False

#- specify optimizer
optimizer = torch.optim.Adam( model.parameters(), lr )
criterion = nn.CrossEntropyLoss().cuda()


#-- load in pre-trained weights if exists
if args.weight_file != '':
    pre_trained_model = torch.load(args.weight_file)

    start_epoch = pre_trained_model['epoch']
    best_prec1 = pre_trained_model['best_prec1']

    model_state = model.state_dict()
    model_state.update( pre_trained_model['state_dict'] )
    model.load_state_dict(model_state)


#-- start training
for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, lr)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    print("switch to train mode")
    model.train()
    reset()
    
    end = time.time()
    counter = 0
    hidden_list = model.init_hidden(batchsize)
    for batch in iterate_data(batchsize, resolution, train_flag = True, require_ori_data=False, block_size=block_size):
        inputs, x_indices, y_indices, z_indices, targets = batch

        # measure data loading time
        data_time.update(time.time() - end)

        targets = targets.reshape(-1)

        """
        # Color jitter
        transform = Compose([ToPILImage(),
                             ColorJitter(brightness=1),
                             ToTensor()])

        for b in range(inputs.shape[0]):
            img = inputs[b, :, :, 3:6]
            img_var = transform(torch.from_numpy(img))
            inputs[b, :, :, 3:6] = img_var.numpy()
        """

        input_var = torch.autograd.Variable( torch.from_numpy( inputs ).cuda(), requires_grad = True )
        target_var = torch.autograd.Variable( torch.from_numpy( targets ).cuda(), requires_grad = False  )
        
        x_indices_var = torch.autograd.Variable( torch.from_numpy( x_indices ).cuda(), requires_grad = False  )
        y_indices_var = torch.autograd.Variable( torch.from_numpy( y_indices ).cuda(), requires_grad = False  )
        z_indices_var = torch.autograd.Variable( torch.from_numpy( z_indices ).cuda(), requires_grad = False  )

        
        # compute output
        hidden_list = repackage_hidden(hidden_list)

        output = model(input_var, x_indices_var, y_indices_var, z_indices_var, hidden_list)
        
        output_reshaped = output.permute(0,2,1,3).contiguous().view(-1, output.size(1))
        
        loss = criterion(output_reshaped, target_var)
    
        # measure accuracy and record loss
        prec1 = accuracy(output_reshaped.data, target_var.data, topk=(1, ))
        prec1[0] = prec1[0].cpu().numpy()[0]
        losses.update(loss.data[0], inputs.shape[0])
        top1.update(prec1[0], inputs.shape[0])


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 100 == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, counter, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


            with open('train_log.txt','a') as f:
                  f.write('Epoch: [{0}][{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(
                           epoch, counter, batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1)   )
    
        counter += 1
    

    # evaluate on validation set
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(13)]
    total_correct_class = [0 for _ in range(13)]

    used_file_names = set([])

    print("switch to evaluate mode")
    model.eval()
    reset()

    end = time.time()
    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    counter = 0
    hidden_list = model.init_hidden(batchsize)
    for batch in iterate_data(batchsize, resolution, train_flag = False, require_ori_data=True, block_size=1.0):
        inputs, x_indices, y_indices, z_indices, targets, inputs_ori = batch
        # measure data loading time
        targets = targets.reshape(-1)
        
        input_var = torch.autograd.Variable( torch.from_numpy( inputs ).cuda(), requires_grad = True )
        target_var = torch.autograd.Variable( torch.from_numpy( targets ).cuda(), requires_grad = False  )
        
        x_indices_var = torch.autograd.Variable( torch.from_numpy( x_indices ).cuda(), requires_grad = False  )
        y_indices_var = torch.autograd.Variable( torch.from_numpy( y_indices ).cuda(), requires_grad = False  )
        z_indices_var = torch.autograd.Variable( torch.from_numpy( z_indices ).cuda(), requires_grad = False  )

        # compute output
        hidden_list = repackage_hidden(hidden_list)

        output = model(input_var, x_indices_var, y_indices_var, z_indices_var, hidden_list)
        output_reshaped = output.permute(0,2,1,3).contiguous().view(-1, output.size(1))
        
        loss = criterion(output_reshaped, target_var)
        # measure accuracy and record loss
        prec1 = accuracy(output_reshaped.data, target_var.data, topk=(1, ))
        prec1[0] = prec1[0].cpu().numpy()[0]
        losses.update(loss.data[0], inputs.shape[0])
        top1.update(prec1[0], inputs.shape[0])
        
        # measure global and average accuracy
        preds = output_reshaped.data.cpu().numpy()
        pred_val = preds.argmax(1)
        correct = np.sum(pred_val == targets)
        total_correct += correct
        total_seen += targets.shape[0]
        for i in range(targets.shape[0]):
            l = targets[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # dump visualizations
        for b in range(inputs_ori.shape[0]):
            pred_label = []
            gt_label = []
            for i in range(inputs_ori.shape[1]):
                idx = b * inputs_ori.shape[1] + i
                pred = pred_val[idx]
                gt = targets[idx]
                #
                gt_classes[gt] += 1
                positive_classes[pred] += 1
                true_positive_classes[gt] += int(gt == pred)

    #---- dump logs
    avg_acc = np.mean( np.array(total_correct_class) / np.array(total_seen_class,dtype=np.float) )
    acc = total_correct / float(total_seen)
    
    print('Epoch {} Val Acc {:.3f}  Avg Acc {:.3f} \t'
          .format(epoch, top1.avg, avg_acc))
    
    with open('test_log.txt','a') as f:
        f.write( 'Epoch {} Val Acc {:.3f} Avg Acc {:.3f}\t '
                .format(epoch, top1.avg,  avg_acc))

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


    prec1 = top1.avg
    
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    }, is_best, filename='models/checkpoint_' + str(epoch) + '.pth.tar')