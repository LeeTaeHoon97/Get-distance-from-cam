from msilib.schema import Shortcut
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
        self.darknet_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),nn.BatchNorm2d(32), nn.MaxPool2d(2, 2))
        self.darknet_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),nn.BatchNorm2d(64),nn.MaxPool2d(2, 2))

        self.darknet_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),nn.BatchNorm2d(128))
        self.darknet_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False),nn.BatchNorm2d(64))
        self.darknet_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),nn.BatchNorm2d(128), nn.MaxPool2d(2, 2))

        self.darknet_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),nn.BatchNorm2d(256))
        self.darknet_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False),nn.BatchNorm2d(128))
        self.darknet_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),nn.BatchNorm2d(256), nn.MaxPool2d(2, 2))
        
        self.darknet_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.darknet_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256))
        self.darknet_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.darknet_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256))
        self.darknet_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))   #이부분에서 shortcut으로도 전달
        self.darknet_maxpool13_2 = nn.Sequential(nn.MaxPool2d(2, 2))
        
        self.darknet_conv14 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        self.darknet_conv15 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512))
        self.darknet_conv16 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        self.darknet_conv17 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512))
        self.darknet_conv18 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        
        self.darknet_conv19 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        self.darknet_conv20 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        
        #Yolo v2
        self.yolov2_conv1 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.yolov2_conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.yolov2_conv3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.yolov2_conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512))
        self.yolov2_conv5 = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False),nn.BatchNorm2d(64))
        
        #Concat , last_conv2의 output은 내가 원하는대로 지정해줘도 되는가? 만약 내가 원하는 값이 x,y,w,h,c, distance , numOfClasses(20)이라면, out_channel은 len(anchors) * (6+numOfClasses)인가?  모델구조는 안건드리고 이부분만 건드리면 되는가?
        self.last_conv1= nn.Sequential(nn.Conv2d(256+1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024))
        self.last_conv2= nn.Sequential(nn.Conv2d(1024,  len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False))
        
    def forward(self,input):
        #darknet19
        output1=self.darknet_conv1(input)
        output1=self.darknet_conv2(output1)
        output1=self.darknet_conv3(output1)
        output1=self.darknet_conv4(output1)
        output1=self.darknet_conv5(output1)
        output1=self.darknet_conv6(output1)
        output1=self.darknet_conv7(output1)
        output1=self.darknet_conv8(output1)
        output1=self.darknet_conv9(output1)
        output1=self.darknet_conv10(output1)
        output1=self.darknet_conv11(output1)
        output1=self.darknet_conv12(output1)
        output1=self.darknet_conv13(output1)

        shortcut=output1
        
        output1=self.darknet_maxpool13_2(output1)
        output1=self.darknet_conv14(output1)
        output1=self.darknet_conv15(output1)
        output1=self.darknet_conv16(output1)
        output1=self.darknet_conv17(output1)
        output1=self.darknet_conv18(output1)
        output1=self.darknet_conv19(output1)
        output1=self.darknet_conv20(output1)

        #yolo v2
        output2=self.yolov2_conv1(shortcut)
        output2=self.yolov2_conv2(output2)
        output2=self.yolov2_conv3(output2)
        output2=self.yolov2_conv4(output2)
        output2=self.yolov2_conv5(output2)      #output2.data.size()= batch, (ch)64 x (h)26 x (w)26

        #batch정보는 train시 사용될 batch size, 지금 정해줄때 batch도 정해줘야 학습때 문제 없음
        batch,ch,h,w=output2.data.size()
        output2=output2.view(batch,ch,h//2,2,w//2,2).contiguous()   #shape = batch 64 13 2 13 2 
        output2=output2.permute(0,1,3,5,2,4).contiguous()           #shape = batch 64 2 2 13 13
        output2=output2.view(batch,-1,h//2,w//2)                    #shape = batch 256 13 13
        
        output=torch.cat((output1,output2),1)
        output=self.last_conv1(output)
        output=self.last_conv2(output)

        return output