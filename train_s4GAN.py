import argparse
import os
import sys
import random
import timeit

import cv2
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab
#from model.deeplabv3p import Res_Deeplab 

from model.discriminator import s4GAN_discriminator
from utils.loss import CrossEntropy2d
from data.voc_dataset import VOCDataSet, VOCGTDataSet
from data import get_loader, get_data_path
from data.augmentations import *

start = timeit.default_timer()

DATA_DIRECTORY = './data/voc_dataset/'
DATA_LIST_PATH = './data/voc_list/train_aug.txt'
CHECKPOINT_DIR = './checkpoints/voc_semi_0_125/'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 21 # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes 
DATASET = 'pascal_voc' #pascal_voc or pascal_context 

SPLIT_ID = None

MODEL = 'DeepLab'
BATCH_SIZE = 8
NUM_STEPS = 40000
SAVE_PRED_EVERY = 5000

INPUT_SIZE = '321,321'
IGNORE_LABEL = 255 # 255 for PASCAL-VOC / -1 for PASCAL-Context / 250 for Cityscapes

RESTORE_FROM = './pretrained_models/resnet101-5d3b4d8f.pth'

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_WORKERS = 4
RANDOM_SEED = 1234

LAMBDA_FM = 0.1
LAMBDA_ST = 1.0
THRESHOLD_ST = 0.6 # 0.6 for PASCAL-VOC/Context / 0.7 for Cityscapes

LABELED_RATIO = None  #0.02 # 1/8 labeled data by default

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset to be used")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of the labeled data to full dataset")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda(gpu)  # Ignore label ??
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

criterion = nn.BCELoss()

def main():
    print (args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)
    
    # load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)
    
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    
    model.train()
    model.cuda(args.gpu)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # init D
    model_D = s4GAN_discriminator(num_classes=args.num_classes, dataset=args.dataset)

    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))

    model_D = torch.nn.DataParallel(model_D).cuda()
    cudnn.benchmark = True    

    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.dataset == 'pascal_voc':    
        train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
        #train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                        #scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
            transform.Normalize([.406, .456, .485], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': 505, 'crop_size': 321}
        #train_dataset = get_segmentation_dataset('pcontext', split='train', mode='train', **data_kwargs)
        data_loader = get_loader('pascal_context')
        data_path = get_data_path('pascal_context') 
        train_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        #train_gt_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        
    elif args.dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([RandomCrop_city((256, 512)), RandomHorizontallyFlip()])
        train_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 
        #train_gt_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if args.labeled_ratio is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)
        
        if args.split_id is not None:
            train_ids = pickle.load(open(args.split_id, 'rb'))
            print('loading train ids from {}'.format(args.split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)
        
        pickle.dump(train_ids, open(os.path.join(args.checkpoint_dir, 'train_voc_split.pkl'), 'wb'))
        
        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=4, pin_memory=True)
        trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=4, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.module.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())


    for i_iter in range(args.num_steps):

        loss_ce_value = 0
        loss_D_value = 0
        loss_fm_value = 0
        loss_S_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train Segmentation Network 
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = Variable(images).cuda(args.gpu)
        pred = interp(model(images))
        loss_ce = loss_calc(pred, labels, args.gpu) # Cross entropy loss for labeled data
        
        #training loss for remaining unlabeled data
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain, _, _, _, _ = batch_remain
        images_remain = Variable(images_remain).cuda(args.gpu)
        pred_remain = interp(model(images_remain))
        
        # concatenate the prediction with the input images
        images_remain = (images_remain-torch.min(images_remain))/(torch.max(images_remain)- torch.min(images_remain))
        #print (pred_remain.size(), images_remain.size())
        pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain), dim=1)
          
        D_out_z, D_out_y_pred = model_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss 
  
        # find predicted segmentation maps above threshold 
        pred_sel, labels_sel, count = find_good_maps(D_out_z, pred_remain) 

        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count > 0 and i_iter > 0:
            loss_st = loss_calc(pred_sel, labels_sel, args.gpu)
        else:
            loss_st = 0.0

        # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt, _, _, _ = batch_gt
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
                
        images_gt = images_gt.cuda()
        images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))
            
        D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1)
        D_out_z_gt , D_out_y_gt = model_D(D_gt_v_cat)
        
        # L1 loss for Feature Matching Loss
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
    
        if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
            loss_S = loss_ce +  args.lambda_fm*loss_fm + args.lambda_st*loss_st 
        else:
            loss_S = loss_ce + args.lambda_fm*loss_fm

        loss_S.backward()
        loss_fm_value+= args.lambda_fm*loss_fm

        loss_ce_value += loss_ce.item()
        loss_S_value += loss_S.item()

        # train D
        for param in model_D.parameters():
            param.requires_grad = True

        # train with pred
        pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.
        
        D_out_z, _ = model_D(pred_cat)
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_z, y_fake_) 

        # train with gt
        D_out_z_gt , _ = model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_z_gt, y_real_)
        
        loss_D = (loss_D_fake + loss_D_real)/2.0
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        print('iter = {0:8d}/{1:8d}, loss_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}'.format(i_iter, args.num_steps, loss_ce_value, loss_fm_value, loss_S_value, loss_D_value))

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint  ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
