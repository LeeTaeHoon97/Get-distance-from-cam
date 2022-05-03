import torch.nn as nn

#본 모델은 13x13의 그리드셀과 5개의 앵커박스, 26개의 클래스로 이루어짐
#S=grid cell , B= anchor box or bounding box, C= num of class
class YoloLoss(nn.Module):
    def __init__(self ,S=13,B=5,C=26):
        super(YoloLoss,self).__init__() #호환을 위해 python 2.0방식으로 선언
        self.S=S
        self.B=B
        self.C=C

        self.lambda_noobj=0.5
        self.lambda_coord=5
        self.lambda_zcoord=0.3        #단안카메라 거리측정에 쓰일 loss balance parm
    
    def forward(self,pred,target):
        




