# EfficientDet with Augmentation 
kaggle reference에서 augmentation으로 데이터를 증가시켜 시도.

---

### 실험 1.
기존의 transform은 그대로 사용하되,
1. rotation 90
2. rotation 180
3. rotation 270
4. vertical flip
5. horizontal flip
을 적용하여 데이터를 6배로 증가시키려 함.

efficientDet은 자체적으로 focal loss를 사용하므로, 일단 클래스 불균형은 고려하지 않음.

→ 과적합 문제 발생. validation loss가 계속 증가.

---



kaggle reference 코드와 거의 동일하고, dataset만 변경할 것이지만 편의를 위해 통째로 복사해 옴.
