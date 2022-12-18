import imutils

from imutils import perspective
from imutils import contours

from difflib import SequenceMatcher
from jamo import h2j, j2hcj
import tensorflow as tf
import cv2 as cv
from PIL import Image
from io import BytesIO
import warnings
import string
from knn_ColorDetection import *
from ocr import ocr
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
my = tf.keras.models.load_model('clss_model/my/efnet_v2_5')
jh = tf.keras.models.load_model('clss_model/jh/efnet_v2_5')
bh = tf.keras.models.load_model('clss_model/bh/efnet_v2_5')
seg = tf.keras.models.load_model('seg_model/DeepLab/DL_fine_tuning_ver3')

img_size = 224


def crop_segd(img):
    """
    input : segmented img size:(224,224), range:(0,1)
    output : cropped img size:(224,224), range:(0,1)
    """
    img = (img * 255).astype(np.uint8)
    origin = img.copy()

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    mor_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # edge detecting
    edged = cv.Canny(mor_img, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)

    cnts = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    # max area
    area = list()
    for c in cnts:
        area.append(cv.contourArea(c))
    idex = (area.index(max(area)))

    # making box
    box = cv.minAreaRect(cnts[idex])
    box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    for i in range(4):
        for j in range(2):
            if box[i, j] < 0:
                box[i, j] = 0

    a = int(min(box[:, 1]))
    b = int(max(box[:, 1]))
    c = int(min(box[:, 0]))
    d = int(max(box[:, 0]))

    crop_img = origin[a:b, c:d]
    crop_img = Image.fromarray(crop_img)

    # img size
    img_size = crop_img.size
    x = img_size[0]
    y = img_size[1]
    size = max(x, y) + 20

    resized_img = Image.new(mode='RGB', size=(size, size), color=(0, 0, 0))
    offset = (round((abs(x - size)) / 2), round((abs(y - size)) / 2))
    resized_img.paste(crop_img, offset)
    resized_img = resized_img.resize((224, 224))
    return np.array(resized_img) / 255.0


def remove_punc(s):
    """
    remove puncuation
    i, I, l 은 1로 변경, 0은 o로 변경
    """
    s = str(s)
    result = s.translate(str.maketrans('iIl0', '111o', string.punctuation))
    result = j2hcj(h2j(result))
    return result.replace(' ','')


class Predict():
    """
    Class for Predict pills
    """
    jh_dict = {3: '정제', 0: '경질캡슐', 2: '연질캡슐', 1: '기타'}
    my_dict = {6: '원형',
               8: '장방형',
               9: '타원형',
               0: '기타',
               4: '삼각형',
               10: '팔각형',
               3: '사각형',
               5: '오각형',
               7: '육각형',
               1: '마름모형',
               2: '반원형'}
    bh_dict = {2: 'x', 1: '-', 0: '+', 3: '기타'}

    def __init__(self, data):
        """
        load models
        input : med DB
        """
        self.data = data
        self.my = my
        self.jh = jh
        self.bh = bh
        self.color_data_path = 'hsv_training.data'

    def predict_img(self, num_result, ocr_model, img=None, path=None):
        """
        predict img
        input : number of result, easyOCR model, img_path or file
        output : predicted features in cat codes:(모양, 제형, 분할선, 색상) and dict(str):(식별문자)
        """
        data = self.data.copy()
        data = data.astype({'MY': 'string', 'JH': 'string', 'BH_F': 'string', 'BH_B': 'string'})
        if not img and not path:
            raise Exception('img or path is needed')
            # img = Image.open(path)
        im_raw = np.array(Image.open(BytesIO(path)).resize((img_size, img_size)), dtype=np.float).reshape(
            (-1, 224, 224, 3)) / 255.0
        mask = seg.predict(im_raw, verbose=0)
        im_seg = np.argmax(mask[0], axis=2).reshape(224, 224, 1) * im_raw[0]
        im = crop_segd(im_seg).reshape(-1, 224, 224, 3)
        pred_my = str(np.argmax(self.my.predict(im, verbose=0)))
        pred_jh = str(np.argmax(self.jh.predict(im, verbose=0)))
        pred_bh = str(np.argmax(self.bh.predict(im, verbose=0)))
        pred_cr = predict_hsv((im[0] * 255).astype(np.uint8), self.color_data_path)
        # pred_cr = predict_hsv(load_image_into_numpy_array(path),self.color_data_path)
        pred_txt = ocr((im[0] * 255).astype(np.uint8), pred_jh, ocr_model)

        # inference
        res = data[data['MY'] == pred_my][data['JH'] == pred_jh][(data['BH_F'] == pred_bh)|(data['BH_B'] == pred_bh)].copy()

        texts = list()
        for t in pred_txt:
            texts.append(remove_punc(t).lower())

        scores = list()
        for r in res.itertuples():
            score = 0
            front = remove_punc(r.TEXT_F).lower()
            back = remove_punc(r.TEXT_B).lower()
            # if pred_cr in r.COLOR:
            #     score += 10
            # for t in texts:
            #     if t in front:
            #         score += 2
            #     elif t in back:
            #         score += 2
            if pred_cr in r.COLOR:
                score += 0.5

            texts_list = list(permutations(texts, len(texts)))
            score_list = list()
            for t in texts_list:
                score_f = SequenceMatcher(None, ''.join(t), front).ratio()
                score_b = SequenceMatcher(None, ''.join(t), back).ratio()
                if (score_f|score_b) == 1:
                    score_list.append(10)
                    break
                score_list.append(max(score_f, score_b))
            scores.append(max(score_list))
        res['score'] = scores

        return res.sort_values(by=['score'], ascending=False)[:num_result]
