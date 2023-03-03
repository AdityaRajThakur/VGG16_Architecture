import torch 
import torch.nn as nn 
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader 
from torchvision import datasets

arch_16 = [64 , 64 , 'M' , 128 , 128 ,'M', 256,256,256,'M',512,512,512,'M',512,512,512,'M']
arch_11 = [64 , 'M' , 128 ,'M',256,256,'M',512,512,'M',512,512,'M']


# len(arch_11)

#vgg16 network architecture
class VGGNet(nn.Module):
  def __init__(self,in_channels = 3 ,num_classes = 100):
    super().__init__()
    self.in_channels = in_channels 
    self.num_classes = num_classes
    self.conv_layers = self.build_conv_layers(arch_16)
    self.fc = nn.Sequential(
        nn.Linear(512 * 7 * 7 , 4096),
        nn.ReLU(),
        nn.Dropout(p = 0.5) ,
        nn.Linear(4096 , 4096),
        nn.ReLU() ,
        nn.Dropout(p = 0.5) ,
        nn.Linear(4096 , num_classes)
    )
  def forward(self,x):
    out = self.conv_layers(x) 
    #print(out.shape)
    out = out.reshape(out.shape[0] , -1 ) 
    #print(out.shape)
    out = self.fc(out) 
    return out 
  def build_conv_layers(self , architecture):
    in_channels = self.in_channels 
    layers = [] 
    for x in architecture: 
      if type(x)==int:
        out_channels = x  
        layers.append(nn.Conv2d(
           in_channels=in_channels ,
           out_channels = out_channels ,
           kernel_size  = (3 ,3 ) ,
           stride = (1 ,1 ) , 
           padding= (1 ,1 )  ,
        ))
        layers.append(nn.BatchNorm2d(x)) 
        layers.append(nn.ReLU())
        in_channels = x  
      elif x =='M':
        layers.append(nn.MaxPool2d(kernel_size = (2 ,2 ) ,stride = (2,2)))
    return nn.Sequential(*layers)

model = VGGNet(in_channels = 3 , num_classes = 100) 
x = torch.randn(1,3 ,224 ,224 )
model(x).shape


