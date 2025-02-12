# [Kaggle] CZII - CryoET Object Identification

이번 포스팅은 Kaggle의 Featured Competition이었던 CZII - CryoET Object Identification 대회 마무리 정리하는 글이다.

- 133/931
- 0.70727(Private), 0.71616(Public)

위와 같은 성적으로 대회를 마무리하였다.

Vision 쪽 경험이 많지 않고, 3D 데이터를 처음 다뤄보았기에 이번 대회가 쉽지 않았지만, 
- 2D UNet을 설계하여 Segmentation을 하고, 후처리를 통해 결과를 얻어내는 방법
- YOLO를 사용하여 Synthetic 데이터로 Pretrian을 한 다음, 대회 데이터로 fine-tuning을 하는 작업, 그리고 전처리하는 과정 등 다양한 방법을 통해 성능을 끌어올리는 과정에서 크게 도움이 되었다.

[Minseok Solution]
- YOLO
    - Dataset을 구성할 때 X, Y Coordinates를 기반으로 box를 생성해야하는데, 2D로 접근하는 경우 대부분 객체의 평균 Radius를 기반으로 설정하였다. 하지만, 객체의 경우 구 형태이기에 ogrid(?)를 통해 각 Slice에서 객체의 범위가 중심 좌표에서 멀어질 수록 작아질 수 있게, 즉 객체의 bbox가 더욱 정확할 수 있게 구성해보았다. (성능 향상 o)
    -> 하지만 평균 radius라는 불확실성으로 인해 Radius를 더 높게 잡았을 때 성능이 좋아지는 것을 확인 가능했다.
    - Loss Function을 대회의 평가 metric에 기반하여 설계하고자 하였지만, 실패
    - YOLO의 Loss Function을 BCE Loss에서 Focal loss로 변경 -> Class Imbalance가 심한 데이터였기에 적합하다고 판단했음. (성능 향상 o)


[1st Place Solution]

해당 솔루션은 Segmentation 모델과 Object Detection 모델을 Ensemble하였음.

[Object Detection]

- Keywords : 
    - Model Architecture : MONAI & SegResNet / DynUNet
    - Data Processing : Patch Extraction
    - Object Detection : Ancher-Free Point Detection, Class Probabilities Map & Offset Map Prediction
    - Loss Function : Varifocal Loss, Point-Point IoU
    - Postprocessing : CenterNet NMS, Top-K Predictions

- 모델 개요
    - Segmentation : 3D UNet 기반 (ResNet & B3 Encoder)
    - Object Detection : SegResNet, DynUNet (Monai 기반)
    - Inference 속도 Optimization : NVidia **TensorRT** 를 통해 Pytorch 대비 200% 속도 향상
        - torch.jit -> ONNX 변환 -> TensorRT 변환
    - T4 GPU 2개를 활용해 병렬 추론


- Modeling 접근 방식
    1. 초기 접근:
        - 3D UNet 기반 SegResNet을 사용한 Heatmap 기반 검출 (Gaussian Peak)
        - 하지만, Segmentation 기반 검출이 모델 용량을 낭비할 가능성이 있어 변경
    2. 최종 접근(Object Detection):
        - YOLO 스타일의 Anchor-Free Point Detection 적용
        - Box Detection 대신 Point Detection 수행
        - IoU 대신 Point-Point IoU (exp(-mse(x,y)/ (2 * radius^2))) 사용
    3. Custom Loss Function
        - PP-YOLO의 Loss Function을 기반으로 설계
        - GT Label을 Top-K Predictions에 매칭
        - Class Map 예측에 Varifical Loss, 거리 예측에는 IoU 기반 Loss 사용


- Model 성능 최적화
    - 입력 해상도를 낮춰 Inference 속도 향상
        - Stride 1 -> Stride 2로 변경 시, 50% 속도 향상
        - Stride 4도 가능하지만, F-beta Score가 Stride 2에서 가장 좋았음 (Trial & Error)
    - 5-Fold Cross Validation 적용
        - 총 7개의 Run을 5(train):2(test)로 구성
        - Sliding Window 기반으로 Validation 수행 (Overlap 적용)
    - Data Augmentation
        - 적용 : Z축 회전(±180도), X/Y축 회전(±10도), Scale Jitter(±5%), X/Y/Z Flip
        - 효과 x : MixUp, Copy-Paste, Random-Erasing (CV 상승, LB 하락)


- Postprocessing (후처리)
    1. CenterNet 스타일 NMS 적용
    2. Top-K Predictions 선택
    3. Class 별 Confidence Threshold 적용
    4. Greedy NMS 수행 (Point Detection 기반)
    5. 좌표를 실제 Angstrom 단위로 변환



- 결론 
    - YOLO 기반의 Point Detection을 활용하여 기존 Segmentation 접근보다 높은 성능 확보
    - Stride 2 적용으로 속도 최적화 + TensorRT 활용으로 Pytorch 대비 200% 속도 향상
    - 5-Fold Ensemble을 통해 성능 향상
    - Mixup, Copy-Paste 등 효과가 없던 방법 또한 모두 테스트하여 최적의 학습 방법 도출



[Segmentation]


-------------------
https://github.com/BloodAxe/Kaggle-2024-CryoET
