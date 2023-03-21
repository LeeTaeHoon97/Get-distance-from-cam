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

## 결론

로컬 환경, colab 환경 모두 ram 및 gpu 부족으로 인해 학습에 어려움이 있어,
5 epoch만 학습하였지만,
학습시 노이즈를 포함한 loss 하향곡선을 보였다.
정상적인 동작 확인

추가적으로, 논문을 참고하여 구현하여 kitti dataset을 사용하였지만, kitti dataset의 resize과정에서 파일의 크기가 작아지고, bbox의 경계가 매우 얇아져, 
좋은 성능을 보이지 못하리라 생각됨.

추가적으로. boackbone network를 다크넷을 구현하여 사용하였으나, pre-trained 모델이 아니였으므로 실제 모델만큼의 좋은 성능을 보이지 못했다고 생각
yolo v2 역시 제로베이스에서 학습하여 좋은 성능을 보이지 못함

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
