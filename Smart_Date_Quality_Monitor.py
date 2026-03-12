import streamlit as st
import requests
import base64
from PIL import Image
import cv2
import numpy as np
import time
import os
from inference_sdk import InferenceHTTPClient

st.set_page_config(
    page_title="مراقب جودة التمور الذكي",
    layout="wide"
)

# -----------------------------
# صورة الهيدر
# -----------------------------

HEADER_IMAGE = "header3.jpeg"

if os.path.exists(HEADER_IMAGE):
    with open(HEADER_IMAGE, "rb") as f:
        img = base64.b64encode(f.read()).decode()
else:
    img = ""

st.markdown(f"""
<div style="
height:260px;
display:flex;
align-items:center;
justify-content:center;
font-size:36px;
font-weight:bold;
color:white;
border-radius:15px;
background:
linear-gradient(to bottom,rgba(0,0,0,0) 50%,rgba(58,49,43,0.9) 100%),
url('data:image/jpeg;base64,{img}');
background-size:cover;
background-position:center;">
مراقب جودة التمور الذكي
</div>
""", unsafe_allow_html=True)

# -----------------------------
# CSS
# -----------------------------

st.markdown("""
<style>

.ai-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0 6px 20px rgba(0,0,0,0.08);
border-left:6px solid #E8C75B;
margin-bottom:20px;
}

.upload-box{
background:white;
padding:35px;
border-radius:20px;
border:2px dashed #E8C75B;
text-align:center;
margin-bottom:40px;
}

.center-title{
text-align:center;
margin-top:30px;
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Roboflow
# -----------------------------

rf_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="C1JPPLBr70pBEnS1eNl8"
)

# -----------------------------
# HuggingFace BLIP
# -----------------------------

HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
}

# -----------------------------
# كشف التمور
# -----------------------------

def detect_dates(image_path):

    result = rf_client.infer(
        image_path,
        model_id="dates-jxszk/1"
    )

    return result.get("predictions", [])

# -----------------------------
# رسم المربعات
# -----------------------------

def draw_boxes(image_path, predictions):

    img = cv2.imread(image_path)

    bad_crops=[]
    good_count=0
    bad_count=0

    for i,p in enumerate(predictions):

        x=int(p["x"])
        y=int(p["y"])
        w=int(p["width"])
        h=int(p["height"])

        label=p["class"]

        x1=int(x-w/2)
        y1=int(y-h/2)
        x2=int(x+w/2)
        y2=int(y+h/2)

        x1=max(0,x1)
        y1=max(0,y1)
        x2=min(img.shape[1],x2)
        y2=min(img.shape[0],y2)

        color=(0,255,0)

        if "bad" in label.lower():

            color=(255,0,0)
            bad_count+=1

            crop=img[y1:y2,x1:x2]

            filename=f"bad_date_{i}.jpg"
            cv2.imwrite(filename,crop)

            bad_crops.append(filename)

        else:
            good_count+=1

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

    return img,bad_crops,good_count,bad_count

# -----------------------------
# تحليل الفساد
# -----------------------------

def analyze_spoilage(image_path):

    with open(image_path,"rb") as f:
        img_bytes=f.read()

    response=requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        data=img_bytes
    )

    if response.status_code!=200:
        return "تعذر تحليل الصورة"

    result=response.json()

    if isinstance(result,list):
        caption=result[0]["generated_text"]
    else:
        caption="صورة تمر غير واضحة"

    return caption

# -----------------------------
# رفع الصورة
# -----------------------------

st.markdown("""
<div class="upload-box">
<h3>📤 ارفع صورة للتمور لتحليل الجودة</h3>
</div>
""", unsafe_allow_html=True)

col1,col2=st.columns(2)

with col1:
    file=st.file_uploader("رفع صورة",type=["jpg","png","jpeg"])

with col2:
    camera=st.camera_input("التقاط صورة")

image=None

if file:
    image=Image.open(file).convert("RGB")

elif camera:
    image=Image.open(camera).convert("RGB")

# -----------------------------
# تحليل الصورة
# -----------------------------

bad_dates=[]

if image:

    IMAGE_PATH="date.jpg"
    image.save(IMAGE_PATH)

    col1,col2=st.columns(2)

    with col1:
        st.image(image,caption="الصورة الأصلية")

    with st.spinner("🤖 الذكاء الاصطناعي يحلل الصورة..."):

        predictions=detect_dates(IMAGE_PATH)

        annotated,bad_dates,good_count,bad_count=draw_boxes(
            IMAGE_PATH,
            predictions
        )

        annotated=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)

    with col2:
        st.image(annotated,caption="نتيجة الكشف")

    st.markdown("<h2 class='center-title'>📊 النتائج</h2>",unsafe_allow_html=True)

    total=good_count+bad_count

    quality=0
    if total>0:
        quality=(good_count/total)*100

    colA,colB,colC=st.columns(3)

    colA.metric("التمور الجيدة",good_count)
    colB.metric("التمور الفاسدة",bad_count)
    colC.metric("جودة المحصول",f"{quality:.1f}%")

# -----------------------------
# تحليل التمور الفاسدة
# -----------------------------

if bad_dates:

    col1,col2,col3=st.columns([1,2,1])

    with col2:
        analyze=st.button("🔍 تحليل التمور الفاسدة",use_container_width=True)

    if analyze:

        for img_path in bad_dates:

            description=analyze_spoilage(img_path)

            st.markdown("### 🧠 تقرير الذكاء الاصطناعي")

            col1,col2=st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="ai-card">
                <h4>الوصف البصري</h4>
                {description}
                </div>
                """,unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="ai-card">
                <h4>نصيحة للمزارعين</h4>
                افحص التمور المصابة وتأكد من ظروف التخزين والرطوبة.
                </div>
                """,unsafe_allow_html=True)
