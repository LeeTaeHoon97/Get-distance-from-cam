from distutils.command.config import config
import torch.nn as nn
import torch
#본 모델은 13x13의 그리드셀과 5개의 앵커박스, 8개의 클래스로 이루어짐
#S=grid cell , B= anchor box or bounding box, C= num of class
class YoloLoss(nn.Module):
    def __init__(self ,S=13,B=5,C=8):
        super(YoloLoss,self).__init__() #호환을 위해 python 2.0
        self.mse=nn.MSELoss(reduction="sum")
        self.ce=nn.CrossEntropyLoss(reduction='sum')
        self.S=S
        self.B=B
        self.C=C

        self.lambda_noobj=0.5
        self.lambda_coord=5
        self.lambda_zcoord=0.3        #단안카메라 거리측정에 쓰일 loss balance param
        
    
    def forward(self, pred, target):
        #pred per grid cell = [class],  [box c score][box(x,y,w,h)] ... ,[box5 c score], [box5] , [distance]  = 8+5*5+1=34
        #target  =객체가 하나뿐인 이미지 한장의 경우 13개의 그리드 셀로 분류된뒤 13*13*14 의 형태를 가짐 [class],[Pc = confidence score],[x], [y], [w], [h],[distance]   =   1+6=7 (class 는 one hot encoding)  
        #이미지 크기는 앵커박스 도입을 위해 416 * 416으로 변경
        #target에는 c score가 들어가야하기때문에 이미지를 13x13으로 grid cell을 나눈다. 이후 target의 중심 좌표가 속한 grid cell값에 1을 준다. 
        #5개의 앵커박스에서 iou각각 산출 한 뒤 최대값을 찾아냄.
        batch_size=pred.data.size(0)
        
        pred = pred.reshape(-1,self.S,self.S,self.C+self.B*5+1)         #pred는 모델이 산출하여나온 값이 flatten 되어있는 상태, 이를 13x13x34의 형태 reshape
        

        #0~7 = classes, 8=box1 c score , 9~12=pos , 13 = box2 c score ...... 28 = box 5 c score, 29~32 = box5 pos, 33= dist 

        iou_b1 = self.iou(pred[..., 9:13],target[...,9:13])                         #target은 gt bbox로 5개의 앵커박스가 존재하는게 아닌 하나의 bbox를 가짐.
        iou_b2 = self.iou(pred[..., 14:18],target[...,9:13])                        #target idx 0~7 = class, 8=Pc, 9~12 = pos , 13 = distance
        iou_b3 = self.iou(pred[..., 19:23],target[...,9:13])
        iou_b4 = self.iou(pred[..., 24:28],target[...,9:13])
        iou_b5 = self.iou(pred[..., 29:33],target[...,9:13])


        ious= torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0),iou_b3.unsqueeze(0),iou_b4.unsqueeze(0),iou_b5.unsqueeze(0)],dim=0)
        
       
        iou_maxes, box_index=torch.max(ious,dim=0)        #bestbox에는 iou값이 가장 큰 bounding box의 index가 저장, iobj(ij) 를 말함.

        exists_box=target[...,8].unsqueeze(3)   #Iobj_i, idx(13,13,8)은 box c score이다. 즉 0이면 존재하지 않고 1이면 존재한다.
                                                # reshape에서 -1,S,S,output 형태로 나눠진 상태인데, 이 중 box c score만 가져오기 위해 unsqueeze(3)을 사용  
                                               
                                                
        bestbox=[pred[..., 8:13],pred[..., 13:18],pred[..., 18:23],pred[..., 23:28],pred[..., 28:33]]        #bestbox는 index(0~4)까지의 박스 인덱스이므로 해당 인덱스에 상응하는 리스트 생성
                                    #해당 박스의 각 elem에는 pred anchor의 c score, x,y,w,h가 들어가 있음.                                    
        
        #Localization Loss
        #Iobj_ij를 명시하지 않은 이유 : bestbox[box_index]를 가져온다는 점에서,box_index의 iobj값이 1이라는걸 시사하기 때문

        values, count = torch.mode(box_index[-1])
        max_iou_idx= torch.mode(values.flatten())[0]
        

        x,y,w,h=target[...,9],target[...,10],target[...,11],target[...,12]
        _, x_hat,y_hat,w_hat,h_hat=bestbox[max_iou_idx][...,0],bestbox[max_iou_idx][...,1],bestbox[max_iou_idx][...,2],bestbox[max_iou_idx][...,3],bestbox[max_iou_idx][...,4]         #에러 발생 시 bestbox의 형태가 어떤형태인지 확인할 필요가 있음, c score를 제외한 x y w h 가 hat변수에 들어감.
        

        localization_loss = exists_box * self.mse(torch.cat((x_hat, y_hat, w_hat, h_hat), dim=-1),
                                          torch.cat((x, y, w, h), dim=-1))
        localization_loss = self.lambda_coord * localization_loss


        # localization_loss = exists_box * self.mse(torch.cat((x_hat, y_hat, torch.sqrt(w_hat), torch.sqrt(h_hat)), dim=-1),
        #                                   torch.cat((x, y, torch.sqrt(w), torch.sqrt(h)), dim=-1))



        # local_loss_part1= self.mse(x_hat,x)+self.mse(y_hat,y)
        
        # # local_loss_part2= self.mse(w_hat,w)+self.mse(h_hat,h)               #no sqrt

        # local_loss_part2= self.mse(torch.sqrt(w_hat),torch.sqrt(w))+self.mse(torch.sqrt(h_hat),torch.sqrt(h))
        
        # local_loss_part2=torch.where(torch.isnan(local_loss_part2), torch.zeros_like(local_loss_part2), local_loss_part2)              #convert nan to 0

        # localization_loss = local_loss_part1+local_loss_part2

        # localization_loss=self.lambda_coord*localization_loss


        # localization_loss=localization_loss*exists_box
        # localization_loss=localization_loss


        #Confidence Loss
        c=target[...,8]
        
        conf_loss=0
        no_conf_loss=0

        # Iobj(ij) and Inoobj(ij)를 명시하지 않은 이유: 존재하든 존재하지 않든 모든경우를 다 계산하기 때문
        for i in range(len(bestbox)):  
            if i == max_iou_idx:  #최대 iou를 가진 anchor box일 경우. ,즉 responsible 또는 Iobj_ij
                c_hat=bestbox[i][...,0]
                conf_loss+=torch.pow((c-c_hat),2)
            else:
                c_hat=bestbox[i][...,0]
                no_conf_loss+=torch.pow((c-c_hat),2)

        confidence_loss=sum(conf_loss) + (self.lambda_noobj * sum(no_conf_loss))

        confidence_loss=confidence_loss.unsqueeze(-1)


        #Classification Loss
        p=target[...,0:8]
        p_hat=pred[...,0:8]

        # p_hat = torch.reshape(p_hat, ( 13, 8))
        # p = torch.reshape(p, ( 13, 8))

        # print('p shape',p.shape)
        # print('p shape',p_hat.shape)


        classification_loss= exists_box * self.mse(p_hat,p)


        #Dsitance Regression Loss
        z=target[...,13]
        z_hat=pred[...,33]
    

        # print("self.lambda_zcoord shape",self.lambda_zcoord.shape)

        distance_regression_loss =  (exists_box * self.mse(z_hat,z))


        distance_regression_loss=self.lambda_zcoord *distance_regression_loss

        loss = localization_loss+ confidence_loss+ classification_loss + distance_regression_loss


        return loss
        # return loss ,localization_loss,confidence_loss,classification_loss,distance_regression_loss
    def iou(self,pred,target):          #yolo dataset은 midpoint 형식의 바운딩박스를 가짐.
                                        #shape = [x,y,w,h]

        #it means,  center x - (width/2) is x start
        pred_box_x1=pred[...,0:1]-(pred[...,2:3]/2)
        pred_box_y1=pred[...,1:2]-(pred[...,3:4]/2)
        pred_box_x2=pred[...,0:1]+(pred[...,2:3]/2)
        pred_box_y2=pred[...,1:2]+(pred[...,3:4]/2)

        
        target_box_x1=target[...,0:1]-(target[...,2:3]/2)
        target_box_y1=target[...,1:2]-(target[...,3:4]/2)
        target_box_x2=target[...,0:1]+(target[...,2:3]/2)
        target_box_y2=target[...,1:2]+(target[...,3:4]/2)
        
        #두 사각형의 겹치는 면적 좌표 구하기
        x1=torch.max(pred_box_x1,target_box_x1)
        y1=torch.max(pred_box_y1,target_box_y1)
        x2=torch.max(pred_box_x2,target_box_x2)
        y2=torch.max(pred_box_y2,target_box_y2)

        intersection=(x2-x1).clamp(0) * (y2-y1).clamp(0)        #torch.clamp(0)을 사용, 만약 두 사각형이 겹치지 않는경우 neg~0사이의 값이 나오므로 이를 0으로 조정 해줌.

        pred_box_area=abs((pred_box_x2-pred_box_x1)*(pred_box_y2-pred_box_y1))
        target_box_area=abs((target_box_x2-target_box_x1)*(target_box_y2-target_box_y1))

        return intersection/(pred_box_area+target_box_area-intersection+ 1e-6)

