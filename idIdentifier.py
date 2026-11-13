import paddleocr
import paddle
from PIL import Image
import numpy as np
import json
import re


class idIdent:
    ocr = paddleocr.PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )

    dataSet = []

    # regex

    toyotaRegex = "TOYOTA"
    firstID = "[A-z0-9]{2}-[A-z0-9]{5}"
    secondID = "[A-z0-9]{5}-[A-z0-9]{4}-[A-z0-9]{2}"
    thirdID = "[A-Z]{3}[0-9]{3}"
    fourthID = "[0-9]{4}-[0-9]{2}"

    regexes = [toyotaRegex, firstID, secondID, thirdID, fourthID]

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

        retList = ["_", "_", "_", "_", "_"]
        confList = [0, 0, 0, 0, 0]
        avgconf = 0

        for data in self.dataSet:
            for index, value in enumerate(data["res"]["rec_scores"]):

                if value < conf:
                    print("continuing")
                    continue

                string = data["res"]["rec_texts"][index]

                useMatch = None
                useIndex = -1

                for index2, regex in enumerate(self.regexes):

                    match = re.search(regex, string)

                    if not match:
                        continue

                    if (useMatch):
                        if (len(useMatch.group()) >= len(match.group())):
                            continue

                    if (len(retList[index2]) < len(match.group())):
                        useMatch = match
                        useIndex = index2

                if useIndex >= 0:
                    retList[useIndex] = useMatch.group()
                    confList[useIndex] = value


        return confList, retList


def main():

    img = Image.open("testImgNum_2.png")

    orc = idIdent()

    orc.parseImg(img)

    conf, retList = orc.retTextwConf(.80)

    print(conf, retList)


if __name__ == "__main__":
    main()
