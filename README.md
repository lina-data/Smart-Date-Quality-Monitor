# Smart Date Quality Monitor

An AI-powered system that analyzes images of dates to detect spoiled fruits and generate quality insights for farmers.

The system combines **computer vision and vision-language models** to detect damaged dates and automatically generate a short report explaining possible spoilage causes and recommendations.

---

## 📌 Project Overview

Date production is a major agricultural sector in many countries. Detecting spoiled dates manually can be slow and inconsistent.

**Smart Date Quality Monitor** helps farmers and producers analyze date fruit images using artificial intelligence to:

- Detect spoiled and healthy dates
- Visualize detections with bounding boxes
- Analyze damaged dates using AI
- Generate an automatic spoilage report

---

## 🚀 Features

- Upload an image or capture it using the camera
- AI detection of good and spoiled dates
- Bounding boxes highlighting detected fruits
- AI-generated spoilage analysis
- Simple and interactive interface built with Streamlit

---

## 🧠 Technologies Used

- Python
- Streamlit
- Roboflow (Object Detection)
- Vision-Language Models
- Pillow
- LM Studio (local AI inference)

---

## ⚙️ How It Works

1. The user uploads or captures an image of dates.
2. The detection model identifies good and spoiled fruits.
3. Spoiled dates are automatically cropped.
4. A vision-language model analyzes the damaged fruit.
5. The system generates a report including:

- Cause of spoilage  
- Type of problem  
- Visible signs  
- Advice for farmers  

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/lina-data/date-quality-inspector.git
cd date-quality-inspector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📂 Available Versions

This project includes **two versions of the system** depending on how the AI model is executed.

---

## 1️⃣ Cloud Version (Streamlit Cloud)

File:

```
Smart_Date_Quality_Monitor.py
```

This version is optimized for **Streamlit Cloud deployment**.

It uses:

- Roboflow API for object detection
- Cloud-based inference
- Lightweight libraries compatible with Streamlit Cloud

Run locally with:

```bash
streamlit run Smart_Date_Quality_Monitor.py
```

This version is suitable for:

- Cloud deployment
- Demonstrations
- Lightweight environments

---

## 2️⃣ Local AI Version (LM Studio)

File:

```
Smart_Date_Quality_Monitor_Local_LMStudio.py
```

This version runs locally and uses a **Vision-Language model through LM Studio** to analyze spoiled dates.

The AI model runs on your local machine instead of the cloud.

### Requirements

- LM Studio installed
- A vision-language model downloaded (example: `llava-phi-3-mini`)

Run with:

```bash
streamlit run Smart_Date_Quality_Monitor_Local_LMStudio.py
```

---

# 🧠 Local AI Setup (LM Studio)

1. Install **LM Studio**
2. Download a vision-language model such as:

```
llava-phi-3-mini
```

3. Start the LM Studio local server.

Default endpoint used by the project:

```
http://localhost:1234/v1/chat/completions
```

---

# 🌱 Use Case

This tool can help:

- Farmers
- Date producers
- Agricultural inspectors
- Food quality control teams

to quickly analyze date fruit quality and detect possible spoilage causes.
