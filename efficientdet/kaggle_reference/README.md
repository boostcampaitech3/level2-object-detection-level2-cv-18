# EfficientDet Kaggle Reference 
kaggle에 올라온 글을 reference로 시도.
> train: https://www.kaggle.com/code/shonenkov/training-efficientdet/notebook
> inference: https://www.kaggle.com/code/shonenkov/inference-efficientdet/notebook

---

### 실험 1.
base code로 시도.
1. epochs = 50 - val loss : / val mAP50 : / 리더보드 :
2. 1에서 epochs + 10 - val loss : / val mAP50 : / 리더보드 :

---

### 실험 2.
transform에서 ToGray의 확률을 0.01 → 0.2로 변경.
transfrom에 Rotate 추가.
```
A.OneOf([
    A.Rotate(limit=[90, 90], p=0.9),
    A.Rotate(limit=[180, 180], p=0.9),
    A.Rotate(limit=[270, 270], p=0.9),
],p=0.8),
```

