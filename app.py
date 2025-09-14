#-*- coding:utf-8 -*-

# YOLOv5 🚀 by Ultralytics
# License: AGPL-3.0 (https://www.gnu.org/licenses/agpl-3.0.html)
# Repository: https://github.com/ultralytics/yolov5


import sys, os
from PIL import Image
#import numpy as np
import math
import streamlit as st
import cv2
from ultralytics import YOLO




image_size = 50


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("シャインマスカット房粒数カウント、収穫時期判定アプリ-ai-app_14")
st.sidebar.write("画像認識モデルを使ってシャインマスカットの房粒数カウント、収穫時期の判定をします。")

st.sidebar.write("")
col1,col2 = st.columns(2)

#with col1:    
img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg","JPG"])
elif img_source == "カメラで撮影":
#with col1:    
    img_file = st.camera_input("カメラで撮影")

#with col2:
try:
 if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        if  img_source != "カメラで撮影":
           #st.image(img, caption="対象の画像", width=280,height =280)
           st.image(img, caption="対象の画像", use_column_width=True)
       
      #st.image(img, caption="対象の画像", width=480)
        st.write("")

        img = img.convert("RGB")
       # img = img.resize((image_size,image_size))
        
        model = YOLO('last.pt')
        #ret = model(img,save=True, conf=0.4, iou=0.1)
        ret = model(img,save=True, conf=0.1, iou=0.4)
        annotated_frame = ret[0].plot(labels=True,conf=True)
        annotated_frame = cv2.cvtColor(annotated_frame , cv2.COLOR_BGR2RGB)
        
    
        # 結果の表示
    #with col2:       
        #st.subheader("判定結果")
        st.subheader("判定結果")
        st.image(annotated_frame, caption='出力画像', width=280) 
        #st.image(annotated_frame, caption='出力画像', width=280,height=280) 
        #st.write(camerapos[y] + "です。")
        #st.write(categories[y] + "です。")

      #****** 2025/03/05
        categories = ret[0].boxes.cls #
        shinemuscat1 = [x for x in categories if x == 0]
        shinemuscat2 = [x for x in categories if x == 1]
        shinemuscat3 = [x for x in categories if x == 2]
        shinemuscat4 = [x for x in categories if x == 3]
        shinemuscat5 = [x for x in categories if x == 4]

        shine1 = 1 * len(shinemuscat1)
        shine2 = 2 * len(shinemuscat2)
        shine3 = 3 * len(shinemuscat3)
        shine4 = 4 * len(shinemuscat4)
        shine5 = 5 * len(shinemuscat5)

        total =len(shinemuscat1) + len(shinemuscat2) + len(shinemuscat3) + len(shinemuscat4) + len(shinemuscat5)
        #evaluation = math.floor((shine1 + shine2 + shine3 + shine4 + shine5)/total )
        evaluation = "{:.2f}".format((shine1 + shine2 + shine3 + shine4 + shine5)/total )
        st.write("## 粒の数:" , total)
        st.write("## 収穫判定:" ,evaluation ) 
        #st.write("## 収穫判定:" ,(shine1 + shine2 + shine3 + shine4 + shine5)/total ) 
except AttributeError:
 st.error("エラー：判定出来ませんでした")  
 #st.write("エラー：判定出来ませんでした")   


