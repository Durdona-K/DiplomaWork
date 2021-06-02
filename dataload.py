import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import numpy as np

def data_create(opt):

    if opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(opt.im_size),
                                        transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10), \
                                            scale=(0.8, 1.2), translate=(0.05, 0.05))],p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        valset = dset.MNIST(root=opt.dataroot, download=True, train=False,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.im_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),
                                   ]))

    if opt.dataset == 'svhn':
        dataset = dset.SVHN(root=opt.dataroot, download=True, split='train',
                            transform=transforms.Compose([transforms.Resize(opt.im_size), transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        valset = dset.SVHN(root=opt.dataroot, download=True, split='test',
                           transform=transforms.Compose([transforms.Resize(opt.im_size), transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    elif opt.dataset in 'imagenet':
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.im_size),
                                       transforms.CenterCrop(opt.im_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    return dataset, valset


def create_dataloader(dataset, valset, opt, val_only=False):
    if not val_only:

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=int(opt.workers), drop_last=True)

    testloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=int(opt.workers))#, drop_last=True)  # check this
    # print(len(dataloader[0]))
    if val_only:
        return testloader
    return dataloader, testloader


def imagenet(opt, dataroot):
    TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(opt.im_size),
    transforms.CenterCrop(opt.im_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

    TRANSFORM_IMG2 = transforms.Compose([
    transforms.Resize(opt.im_size),
    transforms.CenterCrop(opt.im_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

    train_pth = dataroot+'/train'
    test_pth = dataroot+'/val'

    train_data = dset.ImageFolder(root=train_pth, transform=TRANSFORM_IMG)
    train_data_loader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,  num_workers=opt.workers)
    test_data = dset.ImageFolder(root=test_pth, transform=TRANSFORM_IMG2)
    test_data_loader  = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.workers) 

    return train_data_loader, test_data_loader