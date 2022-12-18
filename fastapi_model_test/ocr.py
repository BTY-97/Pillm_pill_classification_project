import easyocr
import cv2
import numpy as np


def ocr(img, form, ocr_model):
    # img = image array
    # form = form of the pill in string cat codes
    # ocr_model = import easyOCR model
    # this function return list of string
    reader = ocr_model

    if img is None:
        raise Exception("Can't read the image")

    if 'PIL' in str(type(img)):
        img = np.array(img)

    try: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass

    kernel = np.ones((15,15), np.uint8)
    if form == ('0' or '2'):
        img = cv2.bilateralFilter(img, -1,5, 5)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                             cv2.THRESH_BINARY, 11, 2)
        # img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=1)
    else:
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(5, 5))
        img = clahe.apply(img)
        img = cv2.fastNlMeansDenoising(img, None, 15, 7, 21)

    detection = reader.readtext(img)

    result = list()
    for detect_text in detection:
        result.append(detect_text[1])

    return result
