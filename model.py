import torch
import torch.nn as nn


class Yolo(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):

        super(Yolo,self).__init__()
        self.num_classes=num_classes
        self.anchors=anchors

        #darknet 19 - 원본의 네트워크에선  활성함수를 사용하지 않음. 사용하는게 데이터 역전파 시 손실율이 줄어들거 같은데 왜 안썼을까?
        self.darknet_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.MaxPool2d(2, 2))
        self.darknet_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),nn.MaxPool2d(2, 2))

        self.darknet_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.darknet_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.darknet_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.MaxPool2d(2, 2))

        self.darknet_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.darknet_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        self.darknet_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.MaxPool2d(2, 2))
        
        self.darknet_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.darknet_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.darknet_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.darknet_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.darknet_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.darknet_maxpool13_2 = nn.Sequential(nn.MaxPool2d(2, 2))
        
        self.darknet_conv14 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False))
        self.darknet_conv15 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False))
        self.darknet_conv16 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False))
        self.darknet_conv17 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False))
        self.darknet_conv18 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False))
        
        self.darknet_conv19 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False))
        self.darknet_conv20 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False))
        
        #Yolo v2
        self.yolov2_conv1 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        self.yolov2_conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        self.yolov2_conv3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        self.yolov2_conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        self.yolov2_conv5 = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False))
        
        #Concat
        self.last_conv1= nn.Sequential(nn.Conv2d(256+1024, 1024, 3, 1, 1, bias=False))
        self.last_conv2= nn.Sequential(nn.Conv2d(1024,  len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False))
        