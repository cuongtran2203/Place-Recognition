import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

train_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])