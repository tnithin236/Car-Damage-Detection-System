# 🚗 AI Car Damage Detection & Severity Analysis

<img width="1917" height="1021" alt="Screenshot 2026-10-07 205406" src="https://github.com/user-attachments/assets/04462135-f691-435b-8273-0c5afdaf409c" />


An AI-powered **car damage detection system** built with **YOLO** and **Streamlit**.

The system analyzes vehicle images to detect visible damage, estimate the **damage severity** and **damaged area**, and generate a downloadable inspection report.

> ⚠️ **Note:** Severity and damage-area results are AI-generated estimates and should not be treated as professional insurance, mechanical, or repair assessments.

---

## ✨ Features

* 🔍 **Car Damage Detection** using a YOLO-based model
* 🎯 **Damage Localization** with bounding boxes
* 🧩 **Damage Segmentation** when supported by the trained model
* 📊 **Severity Estimation** using rule-based analysis
* 📐 **Damage Area Estimation** relative to the complete image
* 📄 **Inspection Report Generation**
* ⬇️ **Downloadable Reports**
* 🖥️ **Interactive Streamlit Web Interface**
* 🤖 Support for custom trained YOLO weights

---

## 🛠️ Tech Stack

| Technology | Purpose                  |
| ---------- | ------------------------ |
| Python     | Core development         |
| YOLO       | Vehicle damage detection |
| Streamlit  | Web application          |
| OpenCV     | Image processing         |
| PyTorch    | Deep learning            |
| NumPy      | Numerical operations     |

---

## 📂 Project Structure

```text
Car-Damage-Detection-System/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   └── best.pt             # Trained YOLO model weights
│
├── notebooks/              # Training / experimentation notebooks
│
├── metrics.json            # Training evaluation metrics
│
├── README.md               # Project documentation
│
└── ...                     # Other project files
```

> The exact folder structure may vary depending on your project files.

---

## 📊 Dataset

This project uses the **CarDD (Car Damage Dataset)** dataset.

**Reference:**

> Wang, Li, Wu, *Car Damage Dataset*, IEEE Transactions on Intelligent Transportation Systems, 2023.

The dataset was obtained through Kaggle.

**Important:** Check the dataset license and usage terms before redistributing the dataset or trained weights.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/tnithin236/Car-Damage-Detection-System.git
cd Car-Damage-Detection-System
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Add the Trained Model

Place your trained YOLO weights inside:

```text
models/best.pt
```

The application expects the trained model to be available at this location.

You can also specify a custom model path using:

```bash
DAMAGE_MODEL_PATH=path/to/best.pt streamlit run app.py
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL where you can access the system through your browser.

---

## 🔄 How It Works

```text
              ┌─────────────────┐
              │   Upload Image  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   YOLO Model    │
              │ Damage Detection│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Damage Analysis │
              └────────┬────────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
      ┌──────────────┐   ┌───────────────┐
      │   Severity   │   │ Damage Area   │
      │  Estimation  │   │  Estimation   │
      └──────┬───────┘   └───────┬───────┘
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │ Inspection Report│
              └─────────────────┘
```

---

## 📈 Model Performance

Model performance should be reported using the **actual metrics generated during training**.

For example:

```text
Metrics:
- mAP50: —
- mAP50-95: —
- Precision: —
- Recall: —
```

> Do not enter estimated performance values. Replace the placeholders with the actual values from `metrics.json`.

---

## ⚠️ Limitations

* Severity and damage-area values are **AI estimates**, not insurance or mechanical assessments.
* Damage area is calculated relative to the **whole image**.
* The current system does not perform vehicle segmentation.
* The dataset labels damage types but does not provide detailed component names such as `"front bumper"`.
* The system does not estimate repair costs because there is no labeled repair-cost data.
* Detection performance depends on image quality, camera angle, lighting, and the trained model.

---

## 🔮 Future Improvements

Potential improvements include:

* 🚘 Vehicle/component segmentation
* 🔧 Detection of specific damaged components
* 💰 Repair-cost estimation
* 📱 Mobile-friendly interface
* ☁️ Cloud deployment
* 📊 Improved severity classification
* 🧠 More advanced damage analysis
* 📈 Expanded evaluation and benchmarking

---

## 👨‍💻 Author

**Nithin**

GitHub:
https://github.com/tnithin236

Project Repository:
https://github.com/tnithin236/Car-Damage-Detection-System

---

## 📜 Disclaimer

This project is intended for **educational and research purposes**.

The detected damage, severity, and area measurements are automated estimates and should not be used as a replacement for professional vehicle inspection, insurance assessment, or mechanical evaluation.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
