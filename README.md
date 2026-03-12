# Smart-Date-Quality-Monitor

An AI-powered system that analyzes images of dates to detect spoiled fruits and provide quality insights for farmers.

The project uses **computer vision and vision-language models** to identify damaged or spoiled dates and generate a short report explaining the possible cause of spoilage and recommendations for farmers.

---

## 📌 Project Overview

Date production is a major agricultural sector in many countries. However, detecting spoiled or damaged dates is often done manually, which can be slow and inaccurate.

**Smart Date Quality Inspector** helps farmers and producers quickly analyze date fruit images using artificial intelligence to:

- Detect spoiled dates
- Identify possible causes of spoilage
- Provide recommendations to improve storage and handling

---

## 🚀 Features

- Upload or capture an image of dates
- AI detection of spoiled and healthy dates
- Visual bounding boxes for detected fruits
- AI-generated spoilage analysis report
- Clean and interactive interface using Streamlit

---

## 🧠 Technologies Used

- Python
- Streamlit
- Computer Vision
- Roboflow (Object Detection)
- Vision-Language Models
- OpenCV
- LM Studio (local AI inference)

---

## ⚙️ How It Works

1. Upload an image of dates or capture it using the camera.
2. The detection model identifies good and spoiled dates.
3. Spoiled dates are extracted and analyzed using a vision-language model.
4. The system generates a report including:
   - Cause of spoilage
   - Type of problem
   - Visible signs
   - Advice for farmers

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/date-quality-inspector.git
cd date-quality-inspector
