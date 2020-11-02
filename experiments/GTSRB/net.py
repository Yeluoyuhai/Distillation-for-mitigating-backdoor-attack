import torchvision
from torch import nn

def CNV():
  CNV = torchvision.models.vgg11(pretrained=False)
  CNV.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 43)
  )

  return CNV