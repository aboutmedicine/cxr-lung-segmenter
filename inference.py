import torch
from torchvision import transforms
from utils import get_model, transform_image

model = get_model()

def get_prediction(image_bytes):
    try:
        img = transform_image(image_bytes=image_bytes)
        mask = model.forward(img)
    except Exception:
        return 0, 'error'

    with torch.no_grad():
        y = torch.sigmoid(mask)
        mask = y.squeeze().cpu().numpy()

    return mask > .6
