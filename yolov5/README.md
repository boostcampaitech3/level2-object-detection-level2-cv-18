# Requirements

```
pip install -r requirements.txt
```
# Make name.txt
name.txt 파일을 만들어서 class 이름들을 적는다.
# Train
1. Install Requirements
   ```
   cd yolov5\convert2yolo
   pip install -r requirements.txt
   ```
2. Create Dataset
   ```
   cd yolov5\convert2yolo
   python example.py --datasets COCO --img_path /opt/ml/detection/dataset --label /opt/ml/detection/dataset/train.json --convert_output_path /opt/ml/detection/dataset --img_type ".jpg" --manifest_path /opt/ml/detection/dataset --cls_list_file /opt/ml/detection/dataset/name --cls_list_file /opt/ml/detection/dataset/name.txt
   ```
   
3. Split Dataset
   - install requirements
        ```
        pip install -r requirements
        ```
   - Running
        ```
        $ python cocosplit.py --having-annotations -s 0.8 /path/to/your/coco_annotations.json train.json test.json
        ```
4. Run train.py
   ```
   python train.py --img 1024 --batch 6 --epoch 100 --data custom.yaml --weights yolov5x6.pt --multi-scale
   ```
# Inference

> Run inference.ipynb

> For pseudo labeling 
> ```
> python val.py --weights /path/to/weights/last.pt --data trash.yaml --img 1024 --iou-thres 0.7 --augment --task test --name experiment_name --save-json
> ```

# Pseudo Labeling
> Run pseudo.ipynb

# Re-Train
train.json 대신 pseudo.ipynb를 통해 새롭게 나온 pseudo.json을 통해 다시 위와 같은 과정을 다시 한번 반복하여 학습을 완료한다.