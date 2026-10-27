# 🔍 Image Forgery Detection and Localization

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red?logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Final Year Engineering Project | Image Processing & Computer Vision (IPCV)**
>
> A forensic web application that detects whether a digital image has been tampered with — and **shows exactly WHERE** the forgery is using a heatmap overlay.

---

## 🎯 What It Does

Upload any image and the system:

| Output | Description |
|--------|-------------|
| ✅ **REAL / FAKE verdict** | Tells you if the image is authentic or manipulated |
| 🗺️ **Forgery Localization** | Highlights the tampered region with a red heatmap |
| 📊 **Score Breakdown** | Shows individual scores for ELA, SIFT, and CNN |

---

## ⚙️ Techniques Used

### 1. ELA — Error Level Analysis
JPEG images degrade at a uniform rate when re-saved. Tampered regions were saved separately, so they show different error levels. ELA re-saves the image and amplifies the difference to reveal anomalies.

```
ELA = |Original_pixel − Resaved_pixel| × scale
```

### 2. SIFT — Scale Invariant Feature Transform
Copy-move forgery copies a region from within the same image and pastes it elsewhere. SIFT detects this by extracting feature descriptors and finding spatially separated matches using FLANN + Lowe's ratio test.

### 3. CNN — Convolutional Neural Network
A deep CNN trained on ELA image representations learns to classify images as REAL or FAKE. Architecture: 4 Conv blocks + Dense layers + Sigmoid output.

### 4. Combined Score
```
Final Score = 0.50 × CNN + 0.30 × ELA + 0.20 × SIFT
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/aaashishguptaaa/image-tamper-detection.git
cd image-tamper-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

### 5. Open Browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
image-tamper-detection/
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── README.md
│
├── techniques/
│   ├── ela.py                # Error Level Analysis
│   └── sift.py               # SIFT Copy-Move Detection
│
├── models/
│   └── cnn_model.py          # CNN architecture & prediction
│
├── utils/
│   └── visualization.py      # Heatmap & overlay generation
│
├── templates/
│   ├── index.html            # Main upload + results page
│   └── about.html            # Project info page
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/              # Temporary uploaded images
│
└── dataset/
    ├── real/                 # Place real images here for training
    └── fake/                 # Place forged images here for training
```

---

## 🧠 CNN Training (Optional)

To train the CNN with your own dataset:

1. Place **real images** in `dataset/real/`
2. Place **forged images** in `dataset/fake/`
3. Install TensorFlow: `pip install tensorflow`
4. Run training script (coming soon)

> Without training, ELA + SIFT still work fully for detection.

---

## 📸 Screenshots

| Upload Interface | Results with Heatmap |
|-----------------|---------------------|
| *(Upload your image)* | *(See forgery heatmap)* |

---

## 📚 References

- Farid, H. (2009). *Image Forgery Detection.* IEEE Signal Processing Magazine.
- Lowe, D.G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV.
- Bayar, B. & Stamm, M. (2016). *A Deep Learning Approach to Universal Image Manipulation Detection.*
- Fridrich, J. et al. (2003). *Detection of Copy-Move Forgery in Digital Images.*

---

## 👨‍💻 Author

**Aashish Gupta**
Final Year B.E. | Image Processing & Computer Vision
GitHub: [@aaashishguptaaa](https://github.com/aaashishguptaaa)

---

## 📄 License

This project is licensed under the MIT License.
