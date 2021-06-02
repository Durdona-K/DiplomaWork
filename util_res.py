import numpy as np
import torch
import sys
sys.path.append("..")
import logging
from main.layers import AConv2d, ALinear
from dataload import get_split_cifar100, imagenet50_data, data_create, create_dataloader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from pathlib import Path
### utils for train_res

gpu_boole = torch.cuda.is_available()

def dataset_eval(net, data_loader, loss_, args, print_txt='', verbose = 1, task = 0, round_=False):
    correct = 0
    total = 0
    loss_sum = 0
    if args.dataset == 'pmnist':
        permutations = torch.load('./permutations.pt')
    net.eval()
    for images, labels in data_loader:
        if task>0:
            if args.dataset != 'imagenet50' and args.dataset != 'pmnist':
                for iz, inx in enumerate(range(task*args.n_class, (task+1)*args.n_class)):
                    labels[labels==inx]=iz
        if gpu_boole:
            images = images.cuda()
            # images, labels = images.cuda(), labels.cuda()
        if args.dataset == 'pmnist':
            images = images.view(-1,28*28)[:,permutations[task]]
        outputs = net(images, task = task, round_=round_).cpu()
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()

        loss_sum += loss_(outputs,labels).cpu().data.numpy().item()
        
    correct = np.float(correct)
    total = np.float(total)
    if verbose:
        print(print_txt, 'Accuracy:',(100 * np.float(correct) / np.float(total)), \
            'Loss:', (loss_sum / np.float(total)))

    acc = 100.0 * (np.float(correct) / np.float(total))
    loss = (loss_sum / np.float(total))
    net.train()
    return acc, loss

def all_task_eval(net, task, loss_metric, args, round=False):
    """Evaluate model on all tasks until now. tasks start from 0"""
    Acc, Loss = 0.,0.
    if args.dataset=='pmnist':
        _, testing = data_create(args)
        test_loader = create_dataloader(testing, testing, args, val_only=True)
    for t in range(task+1):
        if args.dataset == 's-cifar100' or args.dataset == 'cifar100':
            _, test_loader = get_split_cifar100(args, t+1, class_size=args.n_class)
        elif args.dataset == 'imagenet50':
            args.dataroot = args.dataroot[:-1] + str(t+1)
            # print("ALL TASK EVAL:",t, args)
            _, test_loader = imagenet50_data(args, args.dataroot[:-1] + str(t+1))
        elif args.dataset=='pmnist': pass
        else:
            raise Exception("Dataset {} loader not available.".format(args.dataset))
        acc, loss = dataset_eval(net, test_loader, loss_=loss_metric, args=args, print_txt='task{}'.format(t), verbose=1, task=t, round_=round)
        Acc+=acc
        Loss+=loss
    return Acc/(task+1), Loss/(task+1)

def imagenet_eval(net, task, loss_metric, args, round=False):
    "All task eval for imagenet50 - bad code "
    TRANSFORM_IMG2 = transforms.Compose([
    transforms.Resize(args.im_size),
    transforms.CenterCrop(args.im_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
    Acc, Loss = 0.,0.
    for t in range(0, task+1):
        if t == 0:
            test_pth = '../data/Imagenet50-Cl/task1/val'
        elif t == 1:
            test_pth = '../data/Imagenet50-Cl/task2/val'
        elif t == 2:
            test_pth = '../data/Imagenet50-Cl/task3/val'
        elif t == 3:
            test_pth = '../data/Imagenet50-Cl/task4/val'
        elif t == 4:
            test_pth = '../data/Imagenet50-Cl/task5/val'
        # test_pth = Path(args.dataroot[:-1]+str(t+1)+'/val')

        test_data = dset.ImageFolder(root=test_pth, transform=TRANSFORM_IMG2)
        test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers) 
        acc, loss = dataset_eval(net, test_data_loader, loss_=loss_metric, args=args, \
            print_txt='task{}'.format(t), verbose=1, task=t, round_=round)
        Acc+=acc
        Loss+=loss
    return Acc/(task+1), Loss/(task+1)

"""Function gets the pruning threshold value of weight*softrnd(adj[task])"""
def adj_mag_prune(model, task, pr_pct=95, type='global'):
    """Assuming _prune has been applied to prune adjacencies based of 
    ''prune_para'' value. For now, using only L1 method to prune after 
    multiplying weight with soft rounded adjacencies. Pruning lowest 
    ''pr_pct'' (0-100) of the magnitudes. Global pruning only for now. 
    Task start from 0"""
    parameters_list = []
    for _,p in model.named_children():
        if isinstance(p, AConv2d) or isinstance(p, ALinear):
            parameters_list += [p]
    t = torch.nn.utils.parameters_to_vector([getattr(p, 'weight')*p.soft_round(getattr(p, 'adjx')[task]) for p in parameters_list])
    threshold = np.percentile(t.abs().cpu().data, pr_pct)
    return threshold


def _prune_global_mag(module, task, threshold):
    """Changes task adj ele to 0 if weight*soft_rnd(adj[task])<=threshold"""
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            mask = (module.weight*module.soft_round(module.adjx[task]) > threshold).data
            print("Params alive:",mask.sum().float()/np.prod(mask.shape))
            l = module.adjx[task]*mask.float()
            module.adjx[task].data.copy_(l.data)
            # A = module.soft_round(module.adjx[task]).byte().float() * module.adjx[task]
            # module.adjx[task].data.copy_(A.data)
            # print("Params alive:",module.soft_round(module.adjx[task]).byte().sum().float()/np.prod(module.adjx[task].shape))
        
    if hasattr(module, 'children'):
        for submodule in module.children():
            _prune_global_mag(submodule, task, threshold)

def prepare_grad_masks(model, task, wt_para=1e-6):
    """Writing this because PyTorch hooks are broken - the 'fixed' weights should be calculated on the fly. However,
    the hook module doesn't allow that, thus, for a new task we save the gradient 'masks' beforehand"""
    if task == 0:
        return None
    grad_masks = {}
    for name, module in model.named_children():
        if isinstance(module, ALinear) or isinstance(module, AConv2d):
            gradient_mask = (module.adjx[0]==0).data
            for k in range(1, task):
                gradient_mask = gradient_mask * (module.adjx[k]==0).data
            gradient_mask = gradient_mask.float()
            grad_masks[name] = gradient_mask
        # fixing BN here as well
        elif isinstance(module, torch.nn.ModuleList): # dependent on how you defined batch norm -not robust needs fixing!
            for k in range(task):
                for bn_name, bn_param in module[k].named_parameters():
                    bn_param.requires_grad=False
                module[k].eval()
        else:
            logging.warn("{} layer doesn't get pruned\n".format(name))
    if len(grad_masks)==0:
        raise Exception("gradient mask list is empty!")
    return grad_masks

def prepare_hooks(model, task, masks, hooks = {}):
    """Hooks for fixing parts of adjacency masked weights"""
    if task == 0:
        return None
    assert(masks is not None)
    for name, module in model.named_children():
        if isinstance(module, ALinear) or isinstance(module, AConv2d):
            m = masks[name]
            hooks[name] = module.weight.register_hook(lambda grad, m=m: grad.mul_(m))
        # fixing BN here as well
        elif isinstance(module, torch.nn.ModuleList): # dependent on how you defined batch norm -not robust needs fixing!
            for k in range(task):
                for bn_name, bn_param in module[k].named_parameters():
                    bn_param.requires_grad=False
                module[k].eval()
        else:
            logging.warn("{} layer doesn't get pruned\n".format(name))
    if len(hooks)==0:
        raise Exception("hook list is empty!")
    return hooks#, gradmask_dict

"""Turn adjacencies and BN off for previous tasks"""
def turn_off_adjx(model, task, bn_off=False):
    logging.debug("Turning adjacencies off")
    for name, module in model.named_children():
        if isinstance(module, ALinear) or isinstance(module, AConv2d):
            module.adjx[task].requires_grad=False
        elif bn_off==True:
            """Turn off batch norm as well for the task/adjx"""
            if isinstance(module, torch.nn.ModuleList): # dependent on how you defined batch norm -not robust needs fixing!
                for bn_name, bn_param in module[task].named_parameters():
                    # print(bn_name)
                    # if isinstance(bn_param, torch.nn.BatchNorm2d):
                    # print("Require_grad=False for {}[{}]".format(name, task))
                    bn_param.requires_grad=False
                module[task].eval()
                # print(module[task+1].training)
    return model

"""Load pretrained adjacency model"""
def load_and_check(args, model, loss_metric):
    assert(args.restart_tsk>0)
    # model.load_state_dict(torch.load(args.model_pth))
    model = torch.load(args.model_pth)
    ## running test check ##
    logging.debug("\nRunning all previous task test before starting new task\n")
    if logging.getLogger().getEffectiveLevel() <=20:
        _,_ = all_task_eval(model, args.restart_tsk-1, loss_metric, args)
    ## fixing adjacencies, and bn again - just to be sure##
    logging.warn("\nTurning old adjx and bn off just to be sure\n")
    for i in range(args.restart_tsk):
        model = turn_off_adjx(model, task=i, bn_off=True)
    return model


def learnable_weights(model, args, verbose=False):
    """Returns %age of weights in the model which can be changed - should confirm in hooks"""
    Total_nodes, Free_nodes = 0., 0.
    for name, module in model.named_children():
        if isinstance(module, ALinear) or isinstance(module, AConv2d):
            assert(module.adjx[0].requires_grad==False)
            # gradient_mask = (module.soft_round(module.adjx[0]).round().float()*\
            #     module.weight.abs()<=args.wt_para).data
            gradient_mask = (module.adjx[0]==0).data
            for k in range(1,args.tasks):
                if module.adjx[k].requires_grad==False:
                    # gradient_mask = gradient_mask * (module.soft_round(module.adjx[k]).round()\
                    #     .float()*module.weight.abs()<=args.wt_para).data
                    gradient_mask = gradient_mask * (module.adjx[k]==0).data
                else:
                    logging.debug("Adjacencies of {} for task <= {} are fixed\n".format(name, k-1))
                    break
            free_lr_nodes = torch.count_nonzero(gradient_mask) * 1.
            total_lr_nodes = torch.numel(gradient_mask) * 1.
            if verbose:
                logging.debug("Free nodes in {} = {}".format(name, free_lr_nodes/total_lr_nodes))
            Free_nodes += free_lr_nodes
            Total_nodes += total_lr_nodes
    return Free_nodes/Total_nodes


def check_model_version(present_model, old_path, task, new_model_pth=False, old_model_pth=True):
    """Check if adjacencies and fixed weights don't change over tasks"""
    assert(task>-1)
    error_signal = False # if true means conditions not met for further training
    logging.debug("\nComparing if adjacencies and fixed weights are same over tasks\n")
    if old_model_pth:
        old_model = torch.load(old_path)
    else: old_model = old_path
    if new_model_pth:
        present_model = torch.load(present_model)

    for (name, module), (old_name, old_module) in zip(present_model.named_children(), old_model.named_children()):
        if isinstance(module, ALinear) or isinstance(module, AConv2d):
            assert(isinstance(old_module, ALinear) or isinstance(old_module, AConv2d))
            if (module.adjx[task] != old_module.adjx[task]).sum().item() != 0:
                logging.ERROR("CAUTION: new {}.adjx.{} are not equal to {}.adjx.{}".format(name, task, old_name, task))
                error_signal = True
            if (module.adjx[task].requires_grad==True) or (old_module.adjx[task].requires_grad==True):
                logging.ERROR("CAUTION: adjx {} requires grad = True".format(task))
                error_signal = True
            ### Checking weights as well
            if (module.soft_round(module.adjx[task]).round()*module.weight != \
                old_module.soft_round(old_module.adjx[task]).round()*old_module.weight).sum().item() != 0:
                logging.ERROR("\nCAUTION: new {} weights are not equal to old {} weights for task {}".\
                    format(name, old_name, task))
                error_signal = True
                # print(module.soft_round(module.adjx[task]).round()*module.weight)
                # print(old_module.soft_round(old_module.adjx[task]).round()*old_module.weight)
                # exit()

        elif isinstance(module, torch.nn.ModuleList):
            assert(isinstance(old_module, torch.nn.ModuleList))
            for (bn_n, bn_p), (old_bn_n, old_bn_p) in zip(module[task].named_parameters(), old_module[task].named_parameters()):
                if (bn_p != old_bn_p).sum().item() != 0:
                    logging.ERROR("CAUTION: new {}.{} are not equal to old {}.{} for task {}".format(name, bn_n, \
                        old_name, old_bn_n, task))
                    error_signal = True
                if (bn_p.requires_grad==True) or (old_bn_p.requires_grad==True):
                    logging.ERROR("CAUTION: {} BN {} requires grad".format(name, task))
                    error_signal = True
            if module[task].training == True:
                logging.ERROR("CAUTION: new model BN {} {} is in train()".format(name, task))
                error_signal = True
            if old_module[task].training == True:
                logging.ERROR("CAUTION: old model BN {} {} is in train()".format(old_name, task))
                error_signal = True
        else:
            logging.warn("{},{} is not being compared!\n".format(name, old_name))

    return error_signal


def change_weights(net, old_model, present_task=0, path=False):
    """change net weights to previous tasks weights where adjx is 1"""
    assert(present_task>0)
    if path:
        old_model = torch.load(old_model)
    for (n, p),(on, op) in zip(net.named_children(), old_model.named_children()):
        if isinstance(p, AConv2d) or isinstance(p, ALinear):
            assert(isinstance(op, AConv2d) or isinstance(op, ALinear))
            mask = (op.adjx[0]>0).data
            for m in range(1, present_task):
                mask += (op.adjx[m]>0).data
            mask[mask>0]=1.
            with torch.no_grad():
                p.weight[mask==1]=op.weight[mask==1]
            
    return net