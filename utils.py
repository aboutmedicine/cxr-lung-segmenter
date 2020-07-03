import io
import base64
import imageio

import torch
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from model import UNet


def get_model():
    PATH = './static/CP_epoch5.pth'
    model = UNet(1)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def generate_png(img_bytes, mask):
    img_array = imageio.imread(img_bytes)
    fig = Figure()
    ax = fig.subplots(1, 2)

    ax[0].set_title('Input image')
    ax[0].imshow(img_array, cmap='gray')
    ax[1].set_title(f'Output mask')
    ax[1].imshow(mask, cmap='gray')
    for a in ax:
        a.set_axis_off()

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return pngImageB64String
