# Get-distance-from-cam
Get distance by using opencv, yolo_v2 for my study

카메라로부터 오브젝트 탐지, 

거리계산 처음엔 카메라 두대를 이용하여 유클리드 거리를 구하는 방식을 생각하였는데, 단안 카메라를 이용한 거리측정에 대한 논문을 발견하여 이를 활용해보기로 하였다.

거리 정보가 포함된 Kitti dataset을 이용하여
distance loss를 추가하여 계산


## Loss
![image](https://user-images.githubusercontent.com/59239082/176442109-bd89b592-96c7-4943-bab7-e195ad40afb3.png)



## disctance loss

![CodeCogsEqn](https://user-images.githubusercontent.com/59239082/209095206-16974f62-44eb-4467-b8b7-756c54750775.png)

![CodeCogsEqn (2)](https://user-images.githubusercontent.com/59239082/177166770-9c536857-bfbe-42f2-a488-e11042404f63.png)

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/59239082/177166821-0dcfb663-0ce1-4b3f-8a97-a2cdb55c67ee.png)

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/59239082/177793899-e5eef728-71ed-45f9-8bbf-6c0f68f1cee3.png)

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/59239082/209095271-ef03ce84-b937-47f2-8602-5f6e2b6644f8.png)


## backward에 관하여

논문 내에서 dataset을 imagenet과 cocodataset을 wordTree로 구성한뒤 Joint training 방식을 통해 
특정 loss에 따라 backward를 다르게 구현합니다만, 이번 프로젝트에서는 Torch의 autograd를 사용하였습니다. 



## 어려웠던 점
### yolo라는 논문이 가진 그리드셀과 바운딩박스의 개념, IOU의 개념이 조금 어려웠다.

### yolo 논문의  loss function에서 사용된 기호 중 B와 S가 직접적으로 명시된 부분이 없어 어려웠다.

### loss function의 sigma(B) 와 sigma(S^2)의 수식은, 시그마를 사용하였으니 코드에 합하는 부분을 구현해줘야 하는데 관련 내용이 없어 어려웠다.
해당 시그마가 의미하는것은 단순 합계가 아닌 객체가 존재하는 그리드셀의 값이였다.

"즉, localization loss, confidence loss는 해당 셀에 실제 객체의 중점이 있을 때 해당 셀에서 출력한 bounding box 중 Ground Truth Box와 IoU가 더높은 bounding box와 Ground Truth Box와의 loss를 계산한 것들입니다.
그리고 classification loss는 해당 셀에 실제 객체의 중점이 있을 때 해당 셀에서 얻은 class score와 label data 사이의 loss를 나타낸 값이죠."

### confidence loss 구현 부분에서 bestbox(responsible)와 나머지 박스들의 c score도 알아야 되는데 max를 통하여 최대값을 찾다보니 모든 박스들의 score값을 알수 없었다.

리스트를 통해 한번에 알아볼수있게 정리하였다.

### Iobj_i (exist)와 Iobj_ij(reponsible)의 차이를 명확히 알지 못하였다.

Iobj_i (exist)는 해당 셀 내에서 객체가 실제로 존재하는지(classification loss에 사용) , 존재시 1, 아닐경우 0


Iobj_ij(responsible)는 스칼라인 confidence score(1(pr_object)*iou(bbox,gt_box))을 np형태로 가공,
 responsible bbox(최대 iou bbox)와 gt box를 비교할것이라는 뜻

### max iou를 가진 pred bbox를 선택하는것이 어려웠다.

max iou 자체는 torch.max를 이용하여 쉽게 찾을 수 있었지만, 이를 bboxes에 매칭하는것이 어려웠다.

(shape is [batchsize, gridcell, gridcell, channel])

unsqueeze를 이용하여 박스 idx 채널을 하나 추가하였고, torch.mode 를 이용하여 most frequency를 파악하여 best box를 선택하였다.

### 학습을 돌려도 colab System memory가 초과해서 종료가 되었다.

사진의 크기는 resize가 되어 416x416의 형태가 되고, 3채널 이미지를 가진다. 배치사이즈는 32로 가정시 즉 예상 이미지당 용량은 416x416x3x32x32bit 인데, 
이는 0.066453504 gb이고 200장 기준 약 13gb가 사용된다. 

메모리의 최대 크기는 12기가인데, 여기서 문제가 발생하는 것이였다.(추가로 Conv2D 혹은 Weighting, bias를 너무 많이 사용하는 모델을 사용시 발생하는 문제였음)

yolo의 모델 크기도 크고, 이미지 사이즈도 큰 편이라 하는 수없이 사진의 갯수를 줄이고 학습 틈틈이  지우면서 학습하였다.

## 결론

할 수 있는 범위 내 가장 큰 범위인 30epoch 학습 진행 (save load를 이용해 두 번에 걸쳐 학습)

loss graph 

<!-- ![image](https://user-images.githubusercontent.com/59239082/228606056-ccf3419c-50ea-4626-bd03-9e5e3184a1d1.png) -->
![image](https://github.com/LeeTaeHoon97/Get-distance-from-cam/assets/59239082/656f9db8-4220-4f22-be6e-ddfc7d24509b)![image](https://github.com/LeeTaeHoon97/Get-distance-from-cam/assets/59239082/14d8e581-b76d-4cfd-8d33-9fedce69f07e)


정상적인 동작 확인

![image](https://github.com/LeeTaeHoon97/Get-distance-from-cam/assets/59239082/8730d63c-0934-40f1-aeae-e3166d05cc6c)
![image](https://github.com/LeeTaeHoon97/Get-distance-from-cam/assets/59239082/a29fe8ca-2cc1-4206-87ae-2b63cc3aad0b)


해당 이미지를 예측한 결과, 세밀한 보정은 아직 진행하지 않았으나 가로 293 ~ 331 영역, 세로 59 ~ 80 영역에서 탐지된 객체가 차로 표현되는 등 정상적인 작동을 확인하였음.


추가적으로, backbone network를 다크넷을 구현하여 사용하였으나, pre-trained 모델이 아니였으므로 실제 모델만큼의 좋은 성능을 보이지 못했다고 생각

yolo v2 역시 제로베이스에서 학습하여 좋은 성능을 보이지 못함

비록 환경여건으로 인해 만족스러운 결과를 얻지는 못하였지만, 학습이 진행된다는 점은 확인되었으므로 어느정도 만족함

추후 기회가 된다면 같은 주제로 프로젝트를 만들고 싶다.



## reference)
### (단안 카메라 논문 링크:http://jkros.org/_common/do.php?a=full&b=33&bidx=2194&aidx=26111)
### (yolov2 : https://arxiv.org/pdf/1612.08242.pdf)
### (번역:https://csm-kr.tistory.com/3)
### yolov1 torch 구현
### https://herbwood.tistory.com/14
### yolov1 tf구현
### https://velog.io/@minkyu4506/YOLO-v1-%EB%A6%AC%EB%B7%B0-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84tensorflow2
### train tolov2 with kitti dataset
### http://yizhouwang.net/blog/2018/07/29/train-yolov2-kitti/
