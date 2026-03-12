import streamlit as st
import requests
import base64
from PIL import Image
import cv2
import time
from inference_sdk import InferenceHTTPClient

# --------------------------------
# إعداد الصفحة
# --------------------------------

st.set_page_config(
    page_title="مراقب جودة التمور الذكي",
    #page_icon="",
    layout="wide"
)

#st.image("/Users/linamac/Desktop/Ai_bootcamp/Final_Project_Lina/header3.jpeg", use_container_width=True)
import base64

# تحويل الصورة إلى base64
with open("/Users/linamac/Desktop/Ai_bootcamp/Final_Project_Lina/header3.jpeg", "rb") as f:
    img = base64.b64encode(f.read()).decode()


st.markdown("""
<style>

.ai-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0 6px 20px rgba(0,0,0,0.08);
border-left:6px solid #E8C75B;
margin-bottom:15px;
}

.ai-card h4{
color:#3A312B;
margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>

.header {{
height:280px;
border-radius:14px;
display:flex;
align-items:center;
justify-content:center;
color:white;
font-size:40px;
font-weight:bold;

background:
linear-gradient(
to bottom,
rgba(0,0,0,0) 0%,
rgba(0,0,0,0) 50%,
rgba(120,80,40,0.55) 75%,
rgba(58,49,43,0.9) 100%
),
url("data:image/jpeg;base64,{img}");

background-size:cover;
background-position:center;
}}

</style>

<div class="header">
 مراقب جودة التمور الذكي
</div>

""", unsafe_allow_html=True)



# --------------------------------
# CSS واجهة حديثة
# --------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(180deg,#F3EFEA,#FFFFFF);
font-family: 'Inter', sans-serif;
}

.hero{
text-align:center;
margin-bottom:40px;
}

.hero h1{
font-size:42px;
color:#3A312B;
}

.hero p{
color:#555;
font-size:18px;
}

.upload-box{
background:white;
padding:40px;
border-radius:20px;
border:2px dashed #E8C75B;
text-align:center;
transition:0.3s;
box-shadow:0 10px 30px rgba(0,0,0,0.05);
margin-bottom:40px;
}

.upload-box:hover{
transform:scale(1.02);
}

.metric-card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 8px 25px rgba(0,0,0,0.06);
text-align:center;
}

.report-card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0 10px 25px rgba(0,0,0,0.08);
border-left:6px solid #E8C75B;
margin-top:20px;
}

.stButton button{
background:#E8C75B;
border:none;
padding:12px 25px;
border-radius:25px;
font-size:16px;
}
.center-title{
text-align:center;
margin-top:40px;
margin-bottom:20px;
}

.center-button{
display:flex;
justify-content:center;
margin-top:20px;
margin-bottom:20px;
}

.center-button button{
width:300px;
font-size:18px;
padding:15px;
border-radius:30px;
background:#E8C75B;
}

[data-testid="stMetric"]{
text-align:center;
}

[data-testid="stMetricLabel"]{
justify-content:center;
font-size:18px;
}

[data-testid="stMetricValue"]{
justify-content:center;
font-size:32px;
}
/* توسيط بطاقات النتائج */
[data-testid="stMetric"]{
display:flex;
flex-direction:column;
align-items:center;
justify-content:center;
text-align:center;
}

/* توسيط عنوان المقياس */
[data-testid="stMetricLabel"]{
justify-content:center !important;
text-align:center;
width:100%;
font-size:18px;
}

/* توسيط الرقم */
[data-testid="stMetricValue"]{
justify-content:center !important;
text-align:center;
width:100%;
font-size:34px;
}
/* توسيط الصور */
.stImage{
display:flex;
justify-content:center;
}

.stImage img{
margin:auto;
display:block;
}
.section-divider{
margin:50px 0;
height:1px;
background:linear-gradient(
to right,
transparent,
#E8C75B,
transparent
);

}
@font-face {
    font-family: 'Almarai', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Almarai', sans-serif;
}




</style>
""", unsafe_allow_html=True)



# --------------------------------
# العنوان
# --------------------------------





# --------------------------------
# Roboflow
# --------------------------------

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="C1JPPLBr70pBEnS1eNl8"
)

# --------------------------------
# LM Studio
# --------------------------------

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "llava-phi-3-mini"

# --------------------------------
# كشف التمور
# --------------------------------

def detect_dates(image_path):

    result = client.infer(
        image_path,
        model_id="dates-jxszk/1"
    )

    return result["predictions"]

# --------------------------------
# رسم المربعات
# --------------------------------

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
        conf=p["confidence"]

        x1=int(x-w/2)
        y1=int(y-h/2)
        x2=int(x+w/2)
        y2=int(y+h/2)

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

        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return img,bad_crops,good_count,bad_count

# --------------------------------
# تحليل الفساد
# --------------------------------

def analyze_spoilage(image_path):

    with open(image_path,"rb") as img:
        base64_image=base64.b64encode(img.read()).decode()

    prompt="""
You are a date fruit disease and quality expert.

Carefully inspect the image of the date fruit.

Identify the possible spoilage based ONLY on what you can see in the image.

Return the report EXACTLY in this format:

Cause of spoilage: ...
Type of problem: ...
Visible signs: ...
Advice for farmers: ...

Important:
Only describe problems that are visually observable in the image.
If the cause is uncertain, say "uncertain".
"""


    payload={
        "model":MODEL_NAME,
        "messages":[
            {
                "role":"user",
                "content":[
                    {"type":"text","text":prompt},
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens":200
    }

    response=requests.post(
        LM_STUDIO_URL,
        headers={"Content-Type":"application/json"},
        json=payload
    )

    result=response.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"]

    return "تعذر تحليل الصورة"

# --------------------------------
# رفع الصورة (Drag & Drop)
# --------------------------------


st.markdown("""
<div class="upload-box">

<h3>📤  ارفع صورة للتمور ليقوم الذكاء الاصطناعي بتحليل الجودة وتقديم تقرير</h3>


</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 📁 رفع صورة")
    file = st.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed")

with col2:
    st.markdown("### 📷 الكاميرا")
    camera = st.camera_input("", label_visibility="collapsed")

if file:
    image = Image.open(file).convert("RGB")

elif camera:
    image = Image.open(camera).convert("RGB")

else:
    image = None



st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True) #####
# --------------------------------
# تحليل الصورة
# --------------------------------

if file:

    image = Image.open(file).convert("RGB")
    image.save("date.jpg")
    
    
    #####
    # left_space, col1, col2, right_space = st.columns([1,2,2,1])


    left_space, col1, col2, right_space = st.columns([1,2,2,1])


    with col1:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.image(image, caption="الصورة الأصلية", width=300)
        st.markdown("</div>", unsafe_allow_html=True)


    with st.spinner("🤖 الذكاء الاصطناعي يقوم بتحليل الصورة..."):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        predictions = detect_dates("date.jpg")

        annotated,bad_dates,good_count,bad_count = draw_boxes("date.jpg",predictions)

        annotated=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)


    with col2:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.image(annotated, caption="نتيجة الكشف", width=300)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)#####
    # --------------------------------
    # النتائج
    # --------------------------------

    st.markdown('<h2 class="center-title">📊 النتائج</h2>', unsafe_allow_html=True)

    total=good_count+bad_count
    quality=0

    if total>0:
        quality=(good_count/total)*100

    colA,colB,colC = st.columns([1,1,1])

    with colA:
        st.metric("التمور الجيدة",good_count)

    with colB:
        st.metric("التمور الفاسدة",bad_count)

    with colC:
        st.metric("جودة المحصول",f"{quality:.1f}%")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)####
    # --------------------------------
    # تحليل الفساد
    # --------------------------------

    if bad_dates:

        col1, col2, col3 = st.columns([1,2,1])

        with col2:
         analyze = st.button("🔍 تحليل التمور الفاسدة", use_container_width=True)

        if analyze:
            

            for img_path in bad_dates:

                report = analyze_spoilage(img_path)

                report = analyze_spoilage(img_path)

cause = ""
problem = ""
signs = ""
advice = ""

for line in report.split("\n"):

    line_lower = line.lower()

    if "cause of spoilage" in line_lower:
        cause = line.split(":",1)[-1].strip()

    elif "type of problem" in line_lower:
        problem = line.split(":",1)[-1].strip()

    elif "visible signs" in line_lower:
        signs = line.split(":",1)[-1].strip()

    elif "advice" in line_lower:
        advice = line.split(":",1)[-1].strip()


st.markdown("### 🧠 تقرير الذكاء الاصطناعي")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="ai-card">
        <h4>سبب الفساد</h4>
        {cause}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="ai-card">
        <h4>نوع المشكلة</h4>
        {problem}
    </div>
    """, unsafe_allow_html=True)


col3, col4 = st.columns(2)

with col3:
    st.markdown(f"""
    <div class="ai-card">
        <h4>العلامات الظاهرة</h4>
        {signs}
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="ai-card">
        <h4>نصيحة للمزارعين</h4>
        {advice}
    </div>
    """, unsafe_allow_html=True)