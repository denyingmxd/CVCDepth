from torchvision.models import resnet50,resnet18,resnet34
import torch
from thop import profile
from thop import clever_format

model = resnet18().cuda()
input = torch.randn(6, 3, 384, 640).cuda()
print(input.requires_grad)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(params)
print(macs)
