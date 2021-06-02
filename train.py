import sys
import logging
# Setting threshold level
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
logging.info("Caution: This is the root logger!")

import torch
import torchvision

sys.path.append("..")
import time
import numpy as np
from layers import _prune

from dataload import imagenet
from res18 import AResNet18  
from util_res import (all_task_eval, change_weights, check_model_version, dataset_eval, imagenet_eval, 
                      learnable_weights, load_and_check, turn_off_adjx)

gpu_boole = torch.cuda.is_available()

import argparse

parser = argparse.ArgumentParser(description='Continual Learning')
parser.add_argument('--name', default='', help='Experiment name')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 64)')
parser.add_argument('--dataroot', default='ImageNet/task1', help='data folder')
parser.add_argument('--dataset', default='imagenet', help='dataset being used')

parser.add_argument('--model', default='ares18', help='model to use: ares18, res18')
parser.add_argument('--n_class', type=int, default=10, help='number of classes')
parser.add_argument('--tasks', default=5, type=int, help='no. of tasks')
parser.add_argument('--im_size', default=32, type=int, help='image dimensions')
parser.add_argument('--cn', default=3, type=int, help='input channels')

parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 2e-3)')
parser.add_argument('--lr_adj', type=float, default=-1, help='adj learning rate')
parser.add_argument('--decay', type=float, default=0, help='adj decay rate')
parser.add_argument('--loss', default='ce', help='loss metric to use; ce=cross entropy')
parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use (default: sgd)')
parser.add_argument('--train_p', type=float, default=95., help='training accuracy value as boundary to stop \
    reducing LR - look in code, -1 value turns it off')

parser.add_argument('--wt_para', default=1e-5, type=float, help='abs() weights under this have their adjx pruned\
    and weight gradients zeroed')

parser.add_argument('--prune_para', default=0.999999, type=float, help='sparsity percentage pruned')
parser.add_argument('--prune_epoch', default=0, type=int, help='prune epoch diff')

parser.add_argument('--workers', default=4, type=int, help='number of process workers')

parser.add_argument('--load_model', default=False, type=bool, help='load a trained model')
parser.add_argument('--restart_tsk', default=0, type=int, help='task to restart with in saved model')
parser.add_argument('--model_pth', default='', help='address of saved model')
parser.add_argument('--turn_off', default=False, type=bool, help='turn off after task')
parser.add_argument('--savename', default='imagenet50_task_', help='name for the saved torch model')
parser.add_argument('--print_ev', default=10, type=int, help='print training statistics at every n epochs')
args = parser.parse_args()

if args.model == 'ares18':
    net = AResNet18(num_classes=args.n_class, tasks=args.tasks, channels=args.cn)
elif args.model == 'res18':
    net = ResNet18(num_classes=args.n_class, channels=args.cn)
    raise Exception("model architecture not found")
# print(list(net.named_buffers()))

if args.loss == 'ce':
    loss_metric = torch.nn.CrossEntropyLoss()

if gpu_boole:
    net = net.cuda()

if args.load_model:
    ### load from saved_model and do the checks, etc. ####
    net = load_and_check(args, net, loss_metric)
    # args.lr=args.lr2

if args.lr_adj == -1:
    args.lr_adj = args.lr
pruned = False

if args.optim == 'adam':
    optimizer = torch.optim.Adam([
                    {'params': (param for name, param in net.named_parameters() if 'adjx' not in name), 'lr':args.lr,'momentum':0},
                    {'params': (param for name, param in net.named_parameters() if 'adjx' in name), 'lr':args.lr_adj,'momentum':0,'weight_decay':args.decay}
                ])
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD([
                {'params': (param for name, param in net.named_parameters() if 'adjx' not in name), 'lr':args.lr,'momentum':0.9,'weight_decay':5e-4},
                {'params': (param for name, param in net.named_parameters() if 'adjx' in name), 'lr':args.lr_adj,'momentum':0,'weight_decay':args.decay}
            ])
else:
    raise Exception("Unknown optimizer error.")

j = args.restart_tsk
if j>0:
    """Hooks are unstable - this is a slow alternative"""
    free_wts = learnable_weights(net, args, verbose=True)
    logging.info("\n Parameters that are still trainable = {} %".format(free_wts*100))

    for ix in range(j):
        """Checking whether old adjx and fixed weights are preserved"""
        error_signal = check_model_version(net,old_path='./{}{}{}.pt'.format(args.name,args.savename,j-1)\
            ,task=ix)
        assert(error_signal==False)

train_loader, test_loader = imagenet(args, pathx)

EPOCHS = list(range(args.epochs))
for epoch in EPOCHS:
            
    t1 = time.time()
    
    print("Task:",j,"- Epoch:",epoch)
    if j>0:
        if logging.getLogger().getEffectiveLevel()<=20:
            print("Test acc for all previous tasks:")
            test_acc, test_loss = imagenet_eval(net, task=j-1, loss_metric=loss_metric, args=args)

        for ix in range(j):
            net = turn_off_adjx(net, ix, bn_off=True)
        
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if gpu_boole:
            x, y = x.cuda(), y.cuda()
            
        outputs = net(x,task=j)
        loss = loss_metric(outputs,y)
        
        loss.backward()
        optimizer.step()
        if j>0:
            net=change_weights(net, old_model='./{}{}{}.pt'.format(args.name,args.savename,j-1), present_task=j, path=True)

        del loss; del x; del y; del outputs

    if (epoch)%args.print_ev==0 or (args.epochs-epoch)<10:
        train_acc, train_loss = dataset_eval(net, train_loader, loss_=loss_metric, args=args, print_txt='Training', verbose = 1, task = j)
        if logging.getLogger().getEffectiveLevel() <=20:
            test_acc, test_loss= dataset_eval(net, test_loader, loss_=loss_metric, args=args, \
                print_txt='Testing', verbose = 1, task = j)
        test_acc_true, test_loss_true= dataset_eval(net, test_loader, loss_=loss_metric, args=args, \
            print_txt='Testing(Binary Mapping)', verbose = 1, task = j, round_=True)
        t2 = time.time()
        logging.debug('Time left for task:',((t2-t1)/60)*(args.epochs-epoch),'minutes')
        print()

    if (epoch >= (args.epochs - args.prune_epoch) and args.epochs>args.prune_epoch):
        _prune(net,task=j,prune_para=args.prune_para, wt_sp=True, wt_para=args.wt_para)
        pruned=True
        if (epoch)%args.print_ev==0 or (args.epochs-epoch)<10:
            if logging.getLogger().getEffectiveLevel()<=20:
                test_acc, test_loss= dataset_eval(net, test_loader, loss_=loss_metric, args=args, \
                    print_txt='Testing after mapping pruning', verbose = 1, task = j)
                test_acc_true, test_loss_true= dataset_eval(net, test_loader, loss_=loss_metric, args=args, \
                    print_txt='Testing(Binary) post prune', verbose = 1, task = j, round_=True)
        if args.train_p != -1:
            if epoch==(args.epochs-11) and epoch<150: # add a argparse for this epoch limit
                if train_acc < args.train_p and args.lr>1e-5: # this value changes for different experiments/datasets/models
                    logging.info("changing LR and adding a few epochs")
                    EPOCHS += list(range(args.epochs, args.epochs+29))
                    args.prune_epoch += 29
                    args.epochs += 29
                    
                    args.lr /= 10.
                    for ip, g in enumerate(optimizer.param_groups):
                        if ip==0: # first parameter group (non adjx)
                            g['lr'] = args.lr
                    logging.debug(optimizer)
        print()

for ix in range(j+1):
    net = turn_off_adjx(net, ix, bn_off=True) # turns off adjacency and BN for the task by requires_grad=False

print("--------------------------------")
print("Test acc for all tasks:")
if logging.getLogger().getEffectiveLevel()<=20:
    test_acc, test_loss = imagenet_eval(net, task=j, loss_metric=loss_metric, args=args)
    print("Test acc, Test loss:",test_acc, test_loss)
test_acc, test_loss = imagenet_eval(net, task=j, loss_metric=loss_metric, args=args, round=True)
print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
print("--------------------------------")
print()
for ix in range(j+1):
    net = turn_off_adjx(net, ix, bn_off=True)
# if j == args.tasks-1:
logging.info("Saving model...")
torch.save(net,'results/{}{}{}.pt'.format(args.name,args.savename,j))
if args.turn_off==True:
    exit(0)
