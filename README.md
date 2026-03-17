# Smart Date Quality Monitor

An AI-powered system that analyzes images of dates to detect spoiled fruits and generate quality insights for farmers.

The system combines **computer vision and vision-language models** to detect damaged dates and automatically generate a short report explaining possible spoilage causes and recommendations.

---

## Project Overview

Date production is a major agricultural sector in many countries. Detecting spoiled dates manually can be slow and inconsistent.

Smart Date Quality Monitor helps farmers and producers analyze date fruit images using artificial intelligence to:

- Detect spoiled and healthy dates
- Visualize detections with bounding boxes
- Analyze damaged dates using AI
- Generate an automatic spoilage report

---

## Features

- Upload an image or capture it using the camera
- AI detection of good and spoiled dates
- Bounding boxes highlighting detected fruits
- AI-generated spoilage analysis
- Simple and interactive interface built with Streamlit

---

## Technologies Used

- Python
- Streamlit
- Roboflow (Object Detection)
- Vision-Language Models
- Pillow
- LM Studio (local AI inference)

---

## How It Works

1. The user uploads or captures an image of dates.
2. The detection model identifies good and spoiled fruits.
3. Spoiled dates are cropped automatically.
4. A vision-language model analyzes the damaged fruit.
5. The system generates a report including:

- Cause of spoilage
- Type of problem
- Visible signs
- Advice for farmers

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/lina-data/date-quality-inspector.git
cd date-quality-inspector
