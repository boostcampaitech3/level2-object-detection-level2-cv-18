# ๐[LEVEL2 P_stage ์ฌํ์ฉ ํ๋ชฉ ๋ถ๋ฅ Object Detection] ์ธ์ ์ค๋ฅด์กฐ
![image](https://user-images.githubusercontent.com/59071505/168442125-cf9bac11-f27d-48ac-a2a3-84c97050b923.png)

&nbsp; 
## ๐ฅ Member ๐ฅ
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
      <a href="https://github.com/kimkihoon0515">๊น๊ธฐํ_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">๊น์น๊ท_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">๋จ๊ถํ_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">์ ์น์ฐ_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">์ฅ์ํธ_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">์กฐ์ ์ง_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 
## ๐Project Overview
๋ฐ์ผํ๋ก ๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ , ์๋น๋๋ ์๋๋ฅผ ์ด๊ณ  ์์ต๋๋ค. ํ์ง๋ง ์ด๋ฌํ ๋ฌธํ๋ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์์ต๋๋ค.

๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค.

๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Detection ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค.

- Input : ์ฐ๋ ๊ธฐ ๊ฐ์ฒด๊ฐ ๋ด๊ธด ์ด๋ฏธ์ง์ bbox ์ ๋ณด(์ขํ, ์นดํ๊ณ ๋ฆฌ,COCO format) 
  
- Output : ๋ชจ๋ธ์ bbox ์ขํ, ์นดํ๊ณ ๋ฆฌ, score ๊ฐ์ ๋ฆฌํดํ์ฌ csv ํ์์ผ๋ก ์ ์ถ

&nbsp; 
## ๐๏ธDataset
- Train Images : 4883 images
- Test Images : 4871 images
- Class Names : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size : 1024x1024

&nbsp;

## ๐งฑStructure
โโโ baseline  
โย ย  โโโ detectron2  
โย ย  โโโ faster_rcnn  
โย ย  โโโ mmdetection  
โย ย  โโโ yolov5  
โย ย  โโโ requirements.txt  
โโโ dataset  
โย ย  โโโ test  
โย ย  โโโ test.json  
โย ย  โโโ train  
โย ย  โโโ train.json  
โโโ mission  
โย ย  โโโ ...  
โโโ sample_submission  
    โโโ faster_rcnn_mmdetection_submission.csv  
    โโโ faster_rcnn_torchvision_submission.csv  
    โโโ submission_ensemble.csv  
    โโโ train_sample.csv  


&nbsp;

## ๐งชExperiments

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

## ๐Result
- ์ด 19 ํ ์ฐธ์ฌ
- Public : 14๋ฑ -> Private : 14๋ฑ
- Public : 0.6912 -> Private : 0.6759
  
![image](https://user-images.githubusercontent.com/59071505/168442374-caeaee8b-39ce-4121-9297-67dad50db8a0.png)

&nbsp;

## ๐กUsage

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

