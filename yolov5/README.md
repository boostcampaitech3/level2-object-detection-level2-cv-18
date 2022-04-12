# Requirements

```
pip install -r requirements.txt
```
# Make name.txt

Since Yolov5 needs txt file which has class names, you should make name.txt format like this
![image](https://user-images.githubusercontent.com/63439911/162967763-ad1ea5d6-c247-475b-80c7-702c494d7d32.png)

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

