# EfficientDet Kaggle Reference 
kaggle에 올라온 글을 reference로 시도.
> train: https://www.kaggle.com/code/shonenkov/training-efficientdet/notebook
> inference: https://www.kaggle.com/code/shonenkov/inference-efficientdet/notebook

---

### 실험 1.
base code로 시도.
1. epochs = 50 - val mAP50 : / 리더보드 :
2. 1에서 epochs + 10 - val mAP50 : 0.4984 / 리더보드 :
3. 2에서 epochs + 15 - val mAP50 : 0.5267 

---

### 실험 2.
1. transform에서 ToGray의 확률을 0.01 → 0.2로 변경.
~30 epochs, val mAP50 : `0.4618` (28 epoch)
loss는 줄었지만 mAP는 좋아지지 않음.

2. transfrom에 Rotate 추가.
```
A.OneOf([
    A.Rotate(limit=[90, 90], p=0.9),
    A.Rotate(limit=[180, 180], p=0.9),
    A.Rotate(limit=[270, 270], p=0.9),
],p=0.8),
```
→ loss가 전혀 줄어들 지 않음.

3. Rotate를 `A.Rotate(limit=[90, 90], p=0.5)` 하나만 둠.
→ 역시 loss가 전혀 줄어들 지 않음.

→ 모델마다 잘 맞는 augmentation이 있는 것 같다.
efficientDet은 Rotation과는 잘 맞지 않는 듯.
reference의 transform을 그대로 사용하기로 함.

---

### 실험 3.
클래스 불균형을 고려하여 증강시킨 데이터로 학습.
1. train set만 학습한 경우.
(validation set에 train set과 augmentation만 다른 같은 이미지가 들어간 것을 깨닫고, validation 값은 의미가 없다고 판단.)
50 epochs, 리더보드 mAP50 : `0.4829`
score threshold를 0.1 → 0.01로 변경하였더니, mAP50 : `0.4721` → `0.4829`

---

### 실험 4.
> TTA: https://www.kaggle.com/code/pohanlin/augmentation-with-efficientdet-d5/notebook
해당 글을 참고하여 TTA 적용.

2 (HorizontalFlip 적용 or None) X 2 (VerticalFlip 적용 or None) X 4 (0, 90, 180, 270도 회전)
= 총 16가지

1. 실험 1.3.에 적용.
iou_threshold=0.44, skip_box_threshold=0.0, score_threshold=0.1 / val mAP = 0.5356
iou_threshold=0.44, skip_box_threshold=0.0, score_threshold=0.0 / val mAP = 0.5106
iou_threshold=0.7, skip_box_threshold=0.3, score_threshold=0.1 / val mAP = 0.5457
**iou_threshold=0.7, skip_box_threshold=0.0, score_threshold=0.1 / val mAP = 0.5457**
iou_threshold=0.5, skip_box_threshold=0.0, score_threshold=0.1 / val mAP = 0.5455

---

### 실험 5.
Semi-Supervised Learning (Pseudo-Labeling)
