# Get-distance-from-cam
Get distance by using opencv for my study

1.실시간 영상에서 카메라로부터 오브젝트 탐지, 

거리계산 처음엔 카메라 두대를 이용하여 유클리드 거리를 구하는 방식을 생각하였는데, 단안 카메라를 이용한 거리측정에 대한 논문을 발견하여 이를 활용해보기로 하였다.

2. x_hat y_hat ,w_hat, h_hat은 각 target값 , 즉 predict 된 bbox의 최대 iou box

3. Loss fucntion 부분 시그마의 S와 B는 각각 셀의 개수와 바운딩 박스의 개수이다.

해야할일) yolo를 이용해 예제 학습해보기(바운딩박스 관련)
loss func 직접구현 해보기
데이터 분리 trainset 7480 -> trainset 6000 , validset 1480
이후 testset 7517




## 어려웠던 점
yolo라는 논문이 가진 그리드셀과 바운딩박스의 개념, IOU의 개념이 조금 어려웠다.

yolo 논문의  loss function에서 사용된 기호 중 B와 S가 직접적으로 명시된 부분이 없어 어려웠다.



## reference)
### (단안 카메라 논문 링크:http://jkros.org/_common/do.php?a=full&b=33&bidx=2194&aidx=26111)
### (yolov2 : https://arxiv.org/pdf/1612.08242.pdf)
### (번역:https://csm-kr.tistory.com/3)
### yolov1 torch 구현
### https://herbwood.tistory.com/14
### yolov1 tf구현
### https://velog.io/@minkyu4506/YOLO-v1-%EB%A6%AC%EB%B7%B0-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84tensorflow2
