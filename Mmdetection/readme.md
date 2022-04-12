# Install Requirements
```
pip install -r requirements.txt
```
# Download Pre-trained Model
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
```
# Train
> Run swinL_train.ipynb

# Inference
> Run swinL_inference.ipynb

# Pseudo Labeling
> Run pseudo_labeling.ipynb

# Re-Train 
pseudo_labeling.ipynb를 통해 새롭게 나온 train_new.json을 swinL 모델로 다시 학습시킨다.