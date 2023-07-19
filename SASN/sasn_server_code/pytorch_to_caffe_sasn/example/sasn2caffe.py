import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
from example.sasn import RoadSeg
import pytorch_to_caffe

def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)






if __name__=='__main__':
    name='SASN'
    net = RoadSeg(num_labels=15)
    state_dict = torch.load("/home/henry/Semester/2023Spring/His/pytorch_to_caffe_sasn/sasn.pth", map_location = 'cpu')
    

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    for key in list(state_dict.keys()):
        patch_instance_norm_state_dict(state_dict, net, key.split('.'))

    net.load_state_dict(state_dict)
    net.eval()
    input=torch.ones([1,3,320,320])
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
