from typing import Union, List
import time
import math
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from predict import *
from sqlalchemy import create_engine, select
from sqlalchemy.schema import MetaData, Table, Column, ForeignKey
import cv2
import uvicorn
import pandas as pd
import easyocr


app = FastAPI()

img_size = 224


# engine = create_engine('sqlite:///pillm_3.db',connect_args={"check_same_thread": False})
# meta = MetaData(engine)
#
# # my = Table('MY',meta,extend_existing=True,autoload=engine)
# # jh = Table('JH',meta,extend_existing=True,autoload=engine)
# # bh = Table('BH',meta,extend_existing=True,autoload=engine)
# med = Table('MED',meta,extend_existing=True,autoload=engine)
# info = Table('INFO',meta,extend_existing=True,autoload=engine)

# pk_num = 5
reader = easyocr.Reader(['ko', 'en'])

my = pd.read_sql_table('MY', 'sqlite:///pillm_3.db')
jh = pd.read_sql_table('JH', 'sqlite:///pillm_3.db')
bh = pd.read_sql_table('BH', 'sqlite:///pillm_3.db')
med = pd.read_sql_table('MED', 'sqlite:///pillm_3.db')
info = pd.read_sql_table('INFO', 'sqlite:///pillm_3.db')


#
@app.post('/prediction')
async def prediction_route(images: List[UploadFile]):
    start = time.time()
    result_list = []
    prediction = pd.DataFrame()
    p = Predict(med)
    for image in images:
        contents = await image.read()
        prediction = pd.concat([prediction, p.predict_img(num_result=5, ocr_model=reader, path=contents)])
    prediction = prediction.sort_values(by=['score'], ascending=False).drop_duplicates(['PK'])
    info_data = pd.merge(left= prediction.PK, right=info, how='left', on='PK')
    shape = pd.merge(left=prediction, right=my.astype({'PK': 'string'}), how='left', left_on='MY', right_on='PK')
    for i in range(len(prediction.PK)):
        result_list.append(
            {'PK':int(prediction.PK.iloc[i]),
             'NAME': info_data.NAME.iloc[i],
             'MY': shape.MY_y.iloc[i],
             'COLOR': prediction.COLOR.iloc[i],
             'IMAGE': info_data.IMG.iloc[i],
             'SCORE': str(prediction.iloc[i].score),
             'EFFECT': info_data.EFFECT.iloc[i],
             'USAGE': info_data.USAGE.iloc[i]}
        )
    end = time.time()
    print(f"{end - start:.5f} sec")
    return result_list
# @app.get("/")
# async def root():
#     query = engine.execute(select(med).where(med.c.PK == pk_num)).fetchone()
#     info_query = engine.execute(select(info).where(info.c.PK == pk_num)).fetchone()
#
#     shape = engine.execute(select(my).where(my.c.PK == query.MY)).fetchone()
#     # select(info).where(info.c.PK == 0)
#     result = [{'PK':query.PK, 'NAME':info_query.NAME, 'MY':shape.MY, 'COLOR':query.COLOR, 'IMAGE':info_query.IMG, 'EFFECT':info_query.EFFECT, 'USAGE':info_query.USAGE}]
#     return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="192.168.0.12", port=9599, reload=True)

