import torch.nn as nn
import torch
#본 모델은 13x13의 그리드셀과 5개의 앵커박스, 8개의 클래스로 이루어짐
#S=grid cell , B= anchor box or bounding box, C= num of class
class YoloLoss(nn.Module):
    def __init__(self ,S=13,B=5,C=8):
        super(YoloLoss,self).__init__() #호환을 위해 python 2.0방식으로 선언
        self.sse=nn.MSELoss(reduction="sum")
        self.S=S
        self.B=B
        self.C=C

        self.lambda_noobj=0.5
        self.lambda_coord=5
        self.lambda_zcoord=0.3        #단안카메라 거리측정에 쓰일 loss balance parm
    
    def forward(self,pred,target):
        #pred per grid cell = [class],  [box c score][box(x,y,w,h)] ... ,[box5 c score], [box5] , [distance]  = 8+5*5+1=34
        #target  =객체가 하나뿐인 이미지 한장의 경우 13개의 그리드 셀로 분류된뒤 13*13*14 의 형태를 가짐 [class],[Pc = confidence score],[x], [y], [w], [h],[distance]   =   8+6=14 (class 는 one hot encoding)  
        #이미지 크기는 448 * 448 로 시작해 앵커박스 도입을 위해 416 * 416으로 변경
        #5개의 앵커박스에서 iou각각 산출 한 뒤 최대값을 찾아냄.
        iou_b1 = self.iou(pred[..., 9:13],target[...,9:13])                         #target은 gt bbox로 5개의 앵커박스가 존재하는게 아닌 하나의 bbox를 가짐.
        iou_b2 = self.iou(pred[..., 14:18],target[...,9:13])                        #target idx 0~7 = class, 8=Pc, 9~12 = pos , 13 = distance
        iou_b3 = self.iou(pred[..., 19:23],target[...,9:13])
        iou_b4 = self.iou(pred[..., 24:28],target[...,9:13])
        iou_b5 = self.iou(pred[..., 29:33],target[...,9:13])
        
        ious= torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0),iou_b3.unsqueeze(0),iou_b4.unsqueeze(0),iou_b5.unsqueeze(0)],dim=0)

        iou_maxes, bestbox=torch.max(ious,dim=0)

        exists_box=target[...,8].unsqueeze(3)   #사진을 grid나누고 분해하여 레이블을 저장할테니, 객체가 있는 부분 과 아닌부분으로 분리됨
                                               
                                                
                                            
        
        box_pred = exists_box*(
            (
                bestbox * pred[...,9:13]
                + (1-bestbox)*pred[]
            )
        )



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

