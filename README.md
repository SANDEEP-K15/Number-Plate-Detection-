Automatic Number Plate Recognition (ANPR) – YOLOv5 + OCR
========================================================

### Overview

This project implements **Automatic Number Plate Recognition (ANPR)** using:

- **YOLOv5-based plate detection** (`ai/ai_model.py`, `models/`, `utils/`)
- **OCR for plate text** using **EasyOCR** (and optional Tesseract) (`ai/ocr_model.py`)
- **Real-time webcam pipeline** (`main.py`)
- **Flask web app** for image-based detection (`app.py`)

You can:
- Run ANPR in **real time** from a webcam and log detected plate numbers.
- Serve a **web interface** where users upload an image and see detected plates drawn on it.

---

## Features

- **YOLOv5-based license plate detector**
- **EasyOCR**-based plate text recognition
- **Filtering & post-processing** of OCR results to keep only plausible plate regions
- **CSV logging** of recognized plate numbers (`ocr_results.csv`)
- **Webcam demo** (`main.py`)
- **Flask web demo** with image upload (`app.py`)

---

## Project Structure

Key files/folders (simplified):

- `main.py` – Real-time ANPR using webcam (YOLOv5 + EasyOCR + CSV logging)
- `app.py` – Flask web app for upload-based detection using YOLOv5
- `ai/`
  - `ai_model.py` – Loads YOLOv5 model and runs detection on frames
  - `ocr_model.py` – EasyOCR / Tesseract helpers
- `helper/`
  - `general_utils.py` – `filter_text`, `save_results` for OCR post-processing & logging
  - `params.py` – ANPR parameters (thresholds, device, model path, etc.)
- `models/`, `utils/` – YOLOv5 model definitions and utilities (ported from Ultralytics)
- `model/`
  - `best.pt`, `last.pt` – Trained YOLOv5 weights for plate detection
- `plate_dataset/` – Example plate images / dataset
- `templates/index.html` – Web UI template
- `static/` – Output images for the Flask app
- `requirements.txt` – Python dependencies

---

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
# From project root: Automatic-number-plate-recognition-YOLO-OCR
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# or CMD
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download / place YOLOv5 weights

- Place your **trained plate detection model** in the `model/` directory:
  - `model/best.pt` or `model/last.pt`

- Ensure the **paths match** the code:

  - For `main.py` / `ai/ai_model.py`, `helper/params.py` currently uses:

    ```python
    self.model = "/home/mef/Documents/plate_detection_project/best.pt"
    ```

    **Update this** to point to your local weights, for example:

    ```python
    self.model = "model/best.pt"
    ```

  - For the Flask app (`app.py`), the model is loaded from:

    ```python
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="model/last.pt", force_reload=True
    )
    ```

    So make sure `model/last.pt` exists, or change the `path` accordingly.

---

## How It Works – High-Level Pipeline

### Real-time ANPR (`main.py`)

1. **Initialize parameters** using `helper/params.py` (`Parameters` class).
2. **Load YOLOv5 model** with `ai/ai_model.py::load_yolov5_model()`.
3. **Open webcam** using OpenCV.
4. **For each frame**:
   - Run **plate detection** via `ai/ai_model.py::detection(frame, model, labels)`.
   - Use **EasyOCR** (`easyocr.Reader`) to read text from the detected plate region.
   - Filter OCR results using `helper/general_utils.py::filter_text(...)`.
   - Append last detected plate text to `ocr_results.csv` via `save_results(...)`.
   - Display frame with bounding boxes and plate label.

### Flask Web App (`app.py`)

1. Load YOLOv5 model via `torch.hub.load(..., path="model/last.pt")`.
2. Provide:
   - `"/"` route:
     - GET: Render upload form (`templates/index.html`).
     - POST: Accept image file, run YOLOv5 detection, save result to `static/image0.jpg`, redirect to it.
   - `"/video"` route:
     - Streams webcam feed with detections in **MJPEG** format.

---

## ANPR Flow Diagram (Mermaid)

```mermaid
flowchart TD
    A[Start] --> B{Entry Point}
    B -->|Real-time mode| C[Run main.py]
    B -->|Web app mode| D[Run app.py]

    %% Real-time ANPR (main.py)
    C --> E[Load Parameters (helper/params.py)]
    E --> F[Load YOLOv5 Model (ai/ai_model.load_yolov5_model)]
    F --> G[Initialize EasyOCR Reader (ai/ocr_model.easyocr_model_load)]
    G --> H[Open Webcam (cv2.VideoCapture)]

    H --> I[Read Frame]
    I --> J[Detect Plate(s) with YOLOv5 (ai/ai_model.detection)]
    J --> K[Crop Plate Region(s)]
    K --> L[Run EasyOCR on Plate (text_reader.readtext)]
    L --> M[Filter OCR Results (helper/general_utils.filter_text)]
    M --> N[Append Last Plate Text to CSV (save_results)]
    N --> O[Show Detection Window (cv2.imshow)]
    O --> P{ESC Pressed?}
    P -->|No| I
    P -->|Yes| Q[Release Camera & Close Windows]
    Q --> R[End]

    %% Web App (app.py)
    D --> S[Load YOLOv5 Model via torch.hub]
    S --> T[/index.html Upload Form/]

    T -->|User Uploads Image| U[Read Image Bytes]
    U --> V[Run YOLOv5 Model (model(img, size=640))]
    V --> W[Render Detection on Image (results.render)]
    W --> X[Save Output as static/image0.jpg]
    X --> Y[Redirect User to Output Image]
    Y --> Z[End Web Request]
```

---

## Usage

### 1. Real-Time Webcam ANPR (`main.py`)

Make sure:

- Your **webcam** is accessible.
- `helper/params.py` points to your YOLO model (`self.model = "model/best.pt"` or similar).

Run:

```bash
python main.py
```

What happens:

- A window opens showing the live camera feed with **detected plates** highlighted.
- The latest plate text is printed in the console.
- Detected plate strings are appended to `ocr_results.csv`.

To exit:

- Press **ESC** in the detection window.

---

### 2. Flask Web App (`app.py`)

Make sure:

- `model/last.pt` exists or `path="model/your_weights.pt"` is set correctly.

Run:

```bash
python app.py
```

By default the app runs on `http://0.0.0.0:5000`.

Steps:

1. Open your browser at `http://localhost:5000`.
2. Upload an image containing a license plate.
3. You’ll be redirected to `static/image0.jpg` showing the detection result.

Additional route:

- `http://localhost:5000/video` – Streams webcam with YOLO detections (no OCR pipeline here).

---

## Configuration Details

Key configuration is in `helper/params.py`:

- **Model & device**
  - `self.model` – Path to YOLO weights (update to match your setup).
  - `self.device` – CUDA if available, else CPU.

- **Inference settings**
  - `self.imgsz` – Input size for the model (default 640).
  - `self.conf_thres` – Confidence threshold for detections.
  - `self.max_det` – Max detections per image.

- **OCR filtering**
  - `self.rect_size` – Normalization factor for bounding box size.
  - `self.region_threshold` – Minimum relative area for a region to be considered valid.

- **Visualization**
  - `self.color`, `self.rect_thickness`, `self.font_scale`, etc.

You can adjust these for your deployment environment or dataset.

---

## Data and Outputs

- **Input examples**
  - `plate_dataset/` – Images of license plates (for testing OCR).
  - `data/images/` – Example images (`bus.jpg`, `zidane.jpg`).

- **Outputs**
  - `ocr_results.csv` – Each run of `main.py` appends recognized plate text.
  - `static/image0.jpg` – Latest output from the Flask app.

---

## Known Notes / Tips

- **Model path**: The hardcoded path in `helper/params.py` is likely from a different environment:
  - `"/home/mef/Documents/plate_detection_project/best.pt"`
  - **You must change this** to a local relative path (e.g., `"model/best.pt"`).

- **Performance**:
  - If running on CPU, inference will be slower; consider reducing `self.imgsz` or `conf_thres`.
  - For faster real-time performance, use a smaller YOLO model (e.g., `yolov5s`-based weights).

- **OCR quality**:
  - EasyOCR performance depends on plate size and image clarity.
  - You may tune `rect_size` and `region_threshold` in `helper/params.py` or pre-process cropping/resizing.

---

## License

This project includes YOLOv5-based components and utilities. See `LICENSE` for full license information and ensure compliance with **Ultralytics YOLOv5** license terms if you redistribute or deploy commercially.

