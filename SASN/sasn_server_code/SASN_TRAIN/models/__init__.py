import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt, dataset):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt, dataset)
    print("model [%s] was created" % (instance.name()))
    return instance



# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
        
#         self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
#         self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         original_out = out
#         out = self.globalAvgPool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         out = out.view(out.size(0), out.size(1), 1, 1)
#         out = out * original_out

#         out += residual
#         out = self.relu(out)

#         return out