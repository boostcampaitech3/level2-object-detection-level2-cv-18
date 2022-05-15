# 🚀[LEVEL2 P_stage 재활용 쓰레기 분류 대회] 언제오르조
![image](https://user-images.githubusercontent.com/59071505/168442125-cf9bac11-f27d-48ac-a2a3-84c97050b923.png)

&nbsp; 
## 🔥 Member 🔥
<table>
  <tr height="125px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515"><img src="https://avatars.githubusercontent.com/kimkihoon0515"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu"><img src="https://avatars.githubusercontent.com/ed-kyu"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo"><img src="https://avatars.githubusercontent.com/GwonPyo"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946"><img src="https://avatars.githubusercontent.com/ysw2946"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551"><img src="https://avatars.githubusercontent.com/jsh0551"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771"><img src="https://avatars.githubusercontent.com/YJ0522771"/></a>
    </td>

  </tr>
  <tr height="70px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515">김기훈_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">김승규_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">남권표_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">유승우_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">장수호_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">조유진_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 
## 🔍Project Overview
바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.

- Input : 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리,COCO format) 
  
- Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴하여 csv 형식으로 제출

&nbsp; 
## 🗂️Dataset
- Train Images : 4883 images
- Test Images : 4871 images
- Class Names : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size : 1024x1024

&nbsp;

## 🧱Structure
├── baseline  
│   ├── detectron2  
│   ├── faster_rcnn  
│   ├── mmdetection  
│   ├── yolov5  
│   └── requirements.txt  
├── dataset  
│   ├── test  
│   ├── test.json  
│   ├── train  
│   └── train.json  
├── mission  
│   └── ...  
└── sample_submission  
    ├── faster_rcnn_mmdetection_submission.csv  
    ├── faster_rcnn_torchvision_submission.csv  
    ├── submission_ensemble.csv  
    └── train_sample.csv  


&nbsp;

## 🧪Experiments

|  | Model | Backbone | Public_mAP | Private_mAP |
| --- | --- | --- | --- | --- |
| 2 Stage | Cascade rcnn | Swin_t | 0.6031 | 0.5858 |
|  | Cascade rcnn | Swin_b | 0.6192 | 0.6066 |
|  | Cascade rcnn | Swin_L | 0.6161 | 0.5964 |
|  | ATSS | Swin_t | 0.5114 | 0.4896 |
| 1Stage | Yolov5x  |  | 0.5652 | 0.5511 |
|  | Yolov5x6  |  | 0.5587 | 0.5329 |
|  | EfficientDet_D5  | EfficientNet | 0.5161 | 0.5077 |
| Ensemble | Cascade + Yolo + EfficienDet |  | 0.6912 | 0.6759 |


&nbsp;

## 🏆Result
- 총 19 팀 참여
- Public : 14등 -> Private : 14등
- Public : 0.6912 -> Private : 0.6759
  
![image](https://user-images.githubusercontent.com/59071505/168442374-caeaee8b-39ce-4121-9297-67dad50db8a0.png)

&nbsp;

## 💡Usage

### Install Requirements
```
pip install -r requirements.txt
```  
### Model Information

[YOLOv5](yolov5)
```
optimizer : SGD
scheduler : lambdaLR
epoch : 70
loss : BCEWithLogitLoss
```

[Swin-L FPN Cascade R-CNN](Mmdetection/Swin_L)
```
optimizedr : AdamW
scheduler : stepLR
epoch : 12
loss : classification : CSE
       bbox : Smooth L1
```

### Ensemble
```
YOLOv5x6(multi-scale+pseudo labeling+TTA+augmentations) + swinL(multi-scale,TTA) + EfficiendDet(offline data,TTA,augmentations)
```

