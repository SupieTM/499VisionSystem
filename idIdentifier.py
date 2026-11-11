import paddleocr
import paddle
from PIL import Image
import numpy as np
import json


class idIdent:
    ocr = paddleocr.PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )

    dataSet = []

    def __init__(self):
        pass

    def parseImg(self, img):

        imgArray = np.array(img.convert("RGB"))

        result = self.ocr.predict(imgArray)

        self.dataSet.clear()

        for res in result:
            data = res.json

            self.dataSet.append(data)

    def readJsons(self):

        for data in self.dataSet:
            print(json.dumps(data, indent=4))

    def readText(self):

        for data in self.dataSet:
            print(data["res"]["rec_texts"])

    def retTextwConf(self, conf):

        retList = []

        for data in self.dataSet:
            for index, value in enumerate(data["res"]["rec_scores"]):

                if value >= conf:
                    retList.append(data["res"]["rec_texts"][index])

        return retList


def main():

    img = Image.open("testImgNum.png")

    orc = idIdent()

    orc.parseImg(img)

    retList = orc.retTextwConf(.80)

    print(retList)


if __name__ == "__main__":
    main()
