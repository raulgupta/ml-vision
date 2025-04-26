
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import os

# Create directory for model
os.makedirs("models/deeplabv3plus", exist_ok=True)

# Load pre-trained DeepLabV3 model
print("Loading pre-trained DeepLabV3 model with ResNet50 backbone...")
model = deeplabv3_resnet50(
    weights='COCO_WITH_VOC_LABELS_V1',
    weights_backbone='IMAGENET1K_V1'
)

# Save the model
model_path = os.path.join("models", "deeplabv3plus", "deeplabv3_resnet50.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save class names
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
    "train", "tvmonitor"
]

class_path = os.path.join("models", "deeplabv3plus", "classes.txt")
with open(class_path, 'w') as f:
    f.write('\n'.join(class_names))
print(f"Class names saved to {class_path}")
