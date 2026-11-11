from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import sys

# Model imports
from transformers import AutoModelForImageSegmentation

import cv2 as cv

ModelName = 'ZhengPeng7/BiRefNet'


class model:
    def __init__(self, name: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForImageSegmentation.from_pretrained(
            name, trust_remote_code=True)

        torch.set_float32_matmul_precision(['high', 'highest'][0])

        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()
        print(f"${name} is ready to use")

        self.imgTrans = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def proccessImg(self, image: Image.Image, thresh=.6):

        # Convert Image to RGB
        image = image.convert("RGB") if image.mode != "RGB" else image

        original_size = image.size

        image = self.imgTrans(image).unsqueeze(
            0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(image)

            prediction = outputs[-1] if isinstance(
                outputs, (list, tuple)
            ) else outputs

            prediction = prediction.sigmoid()

            prediction = torch.nn.functional.interpolate(
                prediction,
                size=(original_size[1], original_size[0]),
                mode = "bilinear",
                align_corners = False
            )


            probability = prediction[0, 0].cpu().numpy()


        #create the mask
        mask = (probability >= thresh).astype(np.uint8) * 255


        return mask, probability



    def parseMask(image, mask, color=(0)):
        resImg = image.convert("RGBA")

        mask = Image.fromarray(mask).convert("L")

        resImg.putalpha(mask)

        return resImg


    def addBg(retImg, size, color):

        retImg = retImg.convert("RGBA")

        bgImg = Image.new("RGBA", size, color)

        return Image.alpha_composite(bgImg, retImg)



def main():

    # Load image using the first command line argument
    img = Image.open(sys.argv[1])

    imgSize = img.size

    # Load model
    mdl = model(ModelName)

    # Proccess image using model
    procMask, procProb = mdl.proccessImg(img)

    retImg = mdl.parseMask(img, procMask, 255)

    retImg = mdl.addBg(retImg, imgSize, (255,255,255))

    retImg.save("test_img.png")

    return 0


if __name__ == '__main__':
    main()
