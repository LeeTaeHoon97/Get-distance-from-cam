import torch
import torch.nn as nn


class Yolo(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):

        super(Yolo,self).__init__()
        self.num_classes=num_classes
        self.anchors=anchors

        self.darknet_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),nn.BatchNorm2d(32), 
                                            nn.LeakyReLU(0.1, inplace=True),nn.MaxPool2d(2, 2))
        self.darknet_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),nn.BatchNorm2d(64),
                                            nn.LeakyReLU(0.1, inplace=True),nn.MaxPool2d(2, 2))

        self.darknet_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),nn.BatchNorm2d(128),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False),nn.BatchNorm2d(64),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),nn.BatchNorm2d(128),
                                            nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))

        self.darknet_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),nn.BatchNorm2d(256),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False),nn.BatchNorm2d(128),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),nn.BatchNorm2d(256),
                                            nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        
        self.darknet_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))   #이부분에서 shortcut으로도 전달
        self.darknet_maxpool13_2 = nn.Sequential(nn.MaxPool2d(2, 2))
        
        self.darknet_conv14 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv15 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv16 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv17 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv18 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        
        self.darknet_conv19 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.darknet_conv20 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        

        
        #Concat 

        # output = {numOfClasses(8),[c(confidence score),x,y,w,h]*numOfAnchors],distance}, out_channel:(len(anchors) * 5)+numOfClasses+1 =5*(5+8+1) = 70 즉, 13x13x70 <- dataset 클래스 종류에 따라 바뀔 수 있음
        self.last_conv1= nn.Sequential(nn.Conv2d(2048+1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                        nn.LeakyReLU(0.1, inplace=True))            #2048 means reorg layer
        self.last_conv2= nn.Sequential(nn.Conv2d(1024,  len(self.anchors) * (5 + self.num_classes+1), 1, 1, 0, bias=False))
    
    

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

        #yolo v2 (reorg)
        def reorg(input_tensor,_stride=2):
            batch_size, channels, height, width = input_tensor.size()
            new_height = height // _stride
            new_width = width // _stride
            
            # 입력을 stride x stride 크기의 블록으로 재배열
            input_tensor = input_tensor.view(batch_size, channels, new_height, _stride, new_width, _stride)
            input_tensor = input_tensor.permute(0, 1, 3, 5, 2, 4)
            input_tensor = input_tensor.contiguous().view(batch_size, channels * _stride * _stride, new_height, new_width)
            return input_tensor

        output2=reorg(shortcut)
         #output2.data.size()= batch_size x (channels * stride * stride) x new_height x new_width = batch * 2048 * 13 * 13


        # #batch정보는 train시 사용될 batch size, 지금 정해줄때 batch도 정해줘야 학습때 문제 없음
        # batch,ch,h,w=output2.data.size()
        # output2=output2.view(batch,ch,h//2,2,w//2,2).contiguous()   #shape = batch 64 13 2 13 2 
        # output2=output2.permute(0,1,3,5,2,4).contiguous()           #shape = batch 64 2 2 13 13
        # output2=output2.view(batch,-1,h//2,w//2)                    #shape = batch 256 13 13
        
        output=torch.cat((output1,output2),1)

        # print("im model's reorg output : ",output.data.size())            # torch.Size([4, 3072, 13, 13])

        output=self.last_conv1(output)
        output=self.last_conv2(output)                              #shape = 8+(5*5)+1, 13, 13
                                                                    #pred per grid cell = [class],  [box c score][box] ... ,[box5 c score], [box5] , [distance] =34
                                                                    # torch.Size([4, 70, 13, 13]) output
        return output