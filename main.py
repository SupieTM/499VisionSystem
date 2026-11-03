from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import sys

# Model imports
from transformers import AutoModelForImageSegmentation

ModelName = 'zhengpeng7/BiRefNet'


class model:
    def __init__(self, name: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForImageSegmentation.from_pretrained(
            name, trust_remote_code=True)

        torch.set_float32_matmul_precision(['high', 'highest'][0])

        self.model.to(self.device)
        self.model.eval()
        print(f"${name} is ready to use")

        self.imgTrans = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def proccessImg(self, image: Image.Image):

        # Convert Image to RGB
        image = image.convert("RGB") if image.mode != "RGB" else image

        pass


def main():

    # Load image using the first command line argument
    img = Image.open(sys.argv[1])

    # Load model
    mdl = model(ModelName)

    # Proccess image using model
    procImg = mdl.proccessImg(img)

    return 0


if __name__ == '__main__':
    main()
