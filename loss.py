import torch.nn as nn
import torch
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
        #pred per grid cell = [class],  [box c score][box(x,y,w,h)] ... ,[box5 c score], [box5] , [distance]  = 20+5*5+1=46
        #5개의 앵커박스에서 iou각각 산출 한 뒤 최대값을 찾아냄.
        iou_b1 = self.iou(pred[..., 21:25],target[...,21:25])
        iou_b2 = self.iou(pred[..., 26:30],target[...,26:30])
        iou_b3 = self.iou(pred[..., 31:35],target[...,31:35])
        iou_b4 = self.iou(pred[..., 36:40],target[...,36:40])
        iou_b5 = self.iou(pred[..., 41:45],target[...,41:45])



    def iou(self,pred,target):          #kitti dataset은 midpoint 형식의 바운딩박스를 가짐.
                                        #shape = [x,y,w,h]
        pred_box_x1=pred[0:1]-pred[2:3]/2
        pred_box_y1=pred[1:2]-pred[3:4]/2
        pred_box_x2=pred[0:1]+pred[2:3]/2
        pred_box_y2=pred[1:2]+pred[3:4]/2

        target_box_x1=target[0:1]-target[2:3]/2
        target_box_y1=target[1:2]-target[3:4]/2
        target_box_x2=target[0:1]+target[2:3]/2
        target_box_y2=target[1:2]+target[3:4]/2
        
        #두 사각형의 겹치는 면적 좌표 구하기
        x1=torch.max(pred_box_x1,target_box_x1)
        y1=torch.max(pred_box_y1,target_box_y1)
        x2=torch.max(pred_box_x2,target_box_x2)
        y2=torch.max(pred_box_y2,target_box_y2)

        intersection=(x2-x1).clamp(0) * (y2-y1).clamp(0)        #torch.clamp(0)을 사용, 만약 두 사각형이 겹치지 않는경우 neg~0사이의 값이 나오므로 이를 0으로 조정 해줌.

        pred_box_area=abs((pred_box_x2-pred_box_x1)*(pred_box_y2-pred_box_y1))
        target_box_area=abs((target_box_x2-target_box_x1)*(target_box_y2-target_box_y1))

        return intersection/(pred_box_area+target_box_area-intersection+ 1e-6)

