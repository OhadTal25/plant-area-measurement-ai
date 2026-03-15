# 🌿 Plant Leaf Area Measurement using AI

AI tool for **automatic plant leaf segmentation and area measurement** from images using **Meta's Segment Anything Model (SAM)**.

This project detects plant leaves in an image, segments them with AI, and calculates their **real-world surface area** using a calibration step.

---

## 🎥 Demo

[![Watch the demo](https://img.shields.io/badge/Watch-Demo-green)](demo.mp4)

The tool allows interactive plant detection and measurement directly from the browser.

---

## 🚀 Features

• Automatic **leaf detection** using Segment Anything
• **Pixel-level segmentation** of plant regions
• **Green color filtering** to isolate vegetation
• **Calibration system** to convert pixels → real-world area
• **Interactive leaf selection**
• **Area measurement per leaf**
• Export results to **CSV**
• Visual overlay of detected regions

---

## 🧠 How It Works

The pipeline:

```
Input Image
      ↓
Segment Anything Model (SAM)
      ↓
Green HSV Filtering
      ↓
Leaf Mask Selection
      ↓
Pixel Area Calculation
      ↓
Calibration
      ↓
Real-World Leaf Area
```

---

## 🛠 Technologies Used

* Python
* PyTorch
* Segment Anything (SAM)
* OpenCV
* NumPy
* Gradio
* Pandas
* Matplotlib

---

## 📦 Installation

Clone the repository:

```
git clone https://github.com/OhadTal25/plant-area-measurement-ai.git
cd plant-area-measurement-ai
```

Install dependencies:

```
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 📥 Download SAM Model

Download the checkpoint:

```
sam_vit_h_4b8939.pth
```

From:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Place it in the project root directory.

---

## ▶️ Run the Application

```
python app.py
```

The Gradio interface will open in your browser.

---

## 📊 Example Output

The system produces:

• segmented plant regions
• area calculation per leaf
• visual overlay
• CSV file with measurements

---

## 🌱 Potential Applications

Plant phenotyping
Agricultural research
Crop growth analysis
Leaf area estimation
Computer vision experiments

---

## 👨‍💻 Author

**Ohad Tal**

AI & Computer Vision enthusiast exploring automation and real-world AI applications.

GitHub:
https://github.com/OhadTal25

---

## ⭐ If you like this project

Give the repository a star ⭐
