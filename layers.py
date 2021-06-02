import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class AConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, datasets=1, same_init=False, Beta=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        self.multi=False
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        if Beta:
            self.Beta = Beta
            self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
            self.initial_beta = self.beta
        else:
            self.Beta = False
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset, round_=False):
        if round_:
            if self.Beta:
                return F.conv2d(input, (self.soft_round(self.adjx[dataset], self.beta).round().float())*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
            return F.conv2d(input, (self.soft_round(self.adjx[dataset]).round().float())*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
            
        if self.Beta:
            return F.conv2d(input, self.soft_round(self.adjx[dataset], self.beta)*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return F.conv2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    def beta_val(self):
        return self.initial_beta.item(), self.beta.item()

class ALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, datasets=1, same_init=False, Beta=False, multi=False):
        super().__init__(in_features, out_features, bias)
        
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        
        self.multi = multi
        if self.multi:
            self.weightx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight),requires_grad=True) for i in range(datasets)])
            for ix in range(datasets):
                self.adjx[ix] = nn.Parameter(torch.ones(*self.adjx[ix].shape),requires_grad=False)


        # if Beta:
        #     self.Beta = Beta
        #     self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
        #     self.initial_beta = self.beta
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset, round_ = False):
        if self.multi:
            weight = self.weightx[dataset]
        else:
            weight = self.weight
            
        if round_:
            try:
                return F.linear(input, (self.soft_round(self.adjx[dataset]).round().float())*weight, self.bias)
            except Exception as e:
                print("DatasetError: {}".format(e))            

        try:
            return F.linear(input, self.soft_round(self.adjx[dataset])*weight, self.bias)
        except Exception as e:
            print("DatasetError: {}".format(e))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    # def beta_val(self):
    #     return self.initial_beta.item(), self.beta.item()
      
class AConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, output_padding=0, datasets=1, same_init=False, Beta=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, output_padding=output_padding)
        
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        
        if Beta:
            self.Beta = Beta
            self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
            self.initial_beta = self.beta
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset):
        if self.Beta:
            return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset], self.beta)*self.weight, bias=None, stride=self.stride, padding=self.padding,
               dilation=self.dilation, output_padding=self.output_padding)
        return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=None, stride=self.stride, padding=self.padding,
               dilation=self.dilation, output_padding=self.output_padding)
        # try:
        #     return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=self.output_padding)
        # except: print("Deconv DatasetError - input shape {}, dataset {}, adjacencies {}".format(input.shape, dataset, len(self.adjx)))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    def beta_val(self):
        return self.initial_beta.item(), self.beta.item()


def _prune(module, task, prune_para, wt_sp=False, wt_para=1e-5, name=None):
    alive, total = 0.0, 0.
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            mask = (module.soft_round(module.adjx[task]) > prune_para).data
            if wt_sp==True:
                """remove adjacencies for small weights as well - wt_para"""
                mask_ = (module.soft_round(module.adjx[task]).float()*module.weight.abs() > wt_para).data
                # mask = torch.logical_and(mask, mask_)
                logging.debug("Mask removes adj for weights less than {} - additional:{}%".format\
                    (wt_para, (1-torch.sum(mask*mask_)/torch.sum(mask))*100.))
                mask = mask * mask_
            logging.debug("{} Adjx [{}] Params alive:{}%".format(name, task, (torch.count_nonzero(mask)*1.)/(1.*torch.numel(mask))*100.))
            l = module.adjx[task]*mask.float()
            if(l.sum().item()==0):
                logging.error("adjx[{}] for {} has no activations".format(task, name))
                if wt_sp == True:
                    mask = (module.soft_round(module.adjx[task]) > prune_para).data
                    mask_ = (module.soft_round(module.adjx[task])*module.weight.abs() > wt_para/10).data
                    logging.debug("Modified Mask removes adj for weights less than {} - additional:{}%".format\
                        (wt_para, torch.sum(mask*mask_)/torch.sum(mask)*100.))
                    mask = mask * mask_
                    l = module.adjx[task]*mask.float()
                else:
                    raise AssertionError("{} adjx[{}] is empty!".format(name, task))
    
    if hasattr(module, 'children'):
        for subname, submodule in module.named_children():
            _prune(submodule, task, prune_para, wt_sp, wt_para, subname)
