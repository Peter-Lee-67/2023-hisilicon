
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F


### help functions ###
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
        net = net
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'pretrained':
                    pass
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    for root_child in net.children():
        for children in root_child.children():
            if children in root_child.need_initialization:
                init_weights(children, init_type, gain=init_gain)
            else:
                init_weights(children, "pretrained", gain=init_gain)
    
    # for root_child in net.children():
    #     #for children in root_child.children():
    #     if root_child in net.need_initialization:
    #         init_weights(root_child, init_type, gain=init_gain)
    #     else:
    #         init_weights(root_child, "pretrained", gain=init_gain)
    return net

def define_RoadSeg(num_labels, init_type='xavier', init_gain=0.02, gpu_ids=[], net_type=1):

    net = RoadSeg(num_labels)
    return init_net(net, init_type, init_gain, gpu_ids)

### network ###
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output



class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch, up_scale=2):
        super(upsample_layer, self).__init__()
        self.up = nn.Upsample(scale_factor=up_scale)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output




class RoadSeg(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self, num_labels):
        super(RoadSeg, self).__init__()

        resnet_raw_model1 = torchvision.models.resnet18(pretrained=True)
        filters = [64, 64, 128, 256, 512]
        
       
        ###  encoder for rgb image  ###
        self.encoder_rgb_conv1 = resnet_raw_model1.conv1
        self.encoder_rgb_bn1 = resnet_raw_model1.bn1
        self.encoder_rgb_relu = resnet_raw_model1.relu
        self.encoder_rgb_maxpool = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.encoder_rgb_layer1 = resnet_raw_model1.layer1
        self.encoder_rgb_layer2 = resnet_raw_model1.layer2
        

        self.conv1 = nn.Conv2d(3, num_labels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_labels)
        self.activation = nn.ReLU(inplace=True)
        
        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up3_0 = upsample_layer(filters[2], filters[1])

        self.final = upsample_layer(filters[0], num_labels)

        ### layers without pretrained model need to be initialized ###
        self.need_initialization = [self.up2_0, self.up3_0,
                                    self.final]

    def forward(self, x1_0):
        x1_0 = self.encoder_rgb_conv1(x1_0)
        x1_0 = self.encoder_rgb_bn1(x1_0)
        x1_0 = self.encoder_rgb_relu(x1_0)
        
        x2_0 = self.encoder_rgb_maxpool(x1_0)
        x2_0 = self.encoder_rgb_layer1(x2_0)

        x3_0 = self.encoder_rgb_layer2(x2_0)


        x2_1 = self.up3_0(x3_0)
        x2_1 = x2_0 + x2_1
        x1_0 = self.up2_0(x2_1)
        
        out = self.final(x1_0)
        return out


class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    def __call__(self, output, target, pixel_average=True):
        if pixel_average:
            return self.loss(output, target).mean() #/ target.data.sum()
        else:
            return self.loss(output, target)




