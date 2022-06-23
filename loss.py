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

        pred = pred.reshape(-1,self.S,self.S,self.C+self.B*5+1)         #pred는 모델이 산출하여나온 값이 flatten 되어있는 상태, 이를 13x13x34의 형태 reshape


        #0~7 = classes, 8=box1 c score , 9~12=pos , 13 = box2 c score ...... 28 = box 5 c score, 29~32 = box5 pos, 33= dist 

        iou_b1 = self.iou(pred[..., 9:13],target[...,9:13])                         #target은 gt bbox로 5개의 앵커박스가 존재하는게 아닌 하나의 bbox를 가짐.
        iou_b2 = self.iou(pred[..., 14:18],target[...,9:13])                        #target idx 0~7 = class, 8=Pc, 9~12 = pos , 13 = distance
        iou_b3 = self.iou(pred[..., 19:23],target[...,9:13])
        iou_b4 = self.iou(pred[..., 24:28],target[...,9:13])
        iou_b5 = self.iou(pred[..., 29:33],target[...,9:13])
        
        ious= torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0),iou_b3.unsqueeze(0),iou_b4.unsqueeze(0),iou_b5.unsqueeze(0)],dim=0)

        iou_maxes, bestbox=torch.max(ious,dim=0)        #bestbox에는 iou값이 가장 큰 bounding box의 index가 저장

        exists_box=target[...,8].unsqueeze(3)   #Iobj_i, idx(8)은 box c score이다. 즉 0이면 존재하지 않고 1이면 존재한다.
                                                # reshape에서 -1,S,S,output 형태로 나눠진 상태인데, 이 중 box c score만 가져오기 위해 unsqueeze(3)을 사용  
                                               
                                                
                                            
        
        #Localization Loss
        x,y,w,h=target[...,9:13]
        x_hat,y_hat,w_hat,h_hat=bestbox         #에러 발생 시bestbox의 형태가 어떤형태인지 확인할 필요가 있음
        local_loss_part1=self.lambda_coord * exists_box * (torch.pow((x-x_hat),2)+torch.pow((y-y_hat),2))
        local_loss_part2=self.lambda_coord * exists_box * (torch.pow((torch.sqrt(w)-torch.sqrt(w_hat)),2)+torch.pow((torch.sqrt(h)-torch.sqrt(h_hat)),2))
        localization_loss = local_loss_part1+local_loss_part2

        #Confidence Loss

        #Classification Loss

        #Dsitance Regression Loss
        z=target[...,14]
        z_hat=pred[...,33]
        distance_regression_loss = self.lambda_zcoord * exists_box * torch.pow((z-z_hat),2)

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

