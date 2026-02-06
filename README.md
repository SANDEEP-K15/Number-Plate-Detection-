# Automatic Number Plate Recognition (ANPR) – YOLOv5 + OCR

### Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** system using:

- **YOLOv5** for license plate detection.
- **EasyOCR** (and optionally Tesseract) for reading plate text.
- A **real-time webcam pipeline** (`main.py`) that detects and reads plates live.
- A **Flask web app** (`app.py`) where users can upload images or view a live detection stream.

The system can:

- Detect vehicle license plates from images or webcam frames.
- Read and **log recognized plate numbers** to a CSV file.
- Provide both **CLI / desktop** and **web-based** interfaces.

---

## Features

- **YOLOv5-based license plate detector**
  - Custom-trained YOLOv5 weights in `model/best.pt` or `model/last.pt`.
- **OCR with EasyOCR**
  - Robust text recognition from cropped plate regions.
- **Post-processing & filtering**
  - Uses geometric filters to keep only plausible plate regions.
- **CSV logging**
  - Saves recognized plate strings into `ocr_results.csv`.
- **Real-time webcam ANPR**
  - Full end-to-end pipeline: detect → crop → OCR → log → display.
- **Flask web app**
  - Upload an image and get a processed image with detections.
  - Optional live MJPEG stream (`/video` route) with YOLOv5 detections.

---

## Project Structure

> (Only important files and directories listed)

- `main.py`  
  Real-time ANPR using webcam (YOLOv5 + EasyOCR + CSV logging).

- `app.py`  
  Flask web app for:
  - `/` – Image upload → detection → output image.
  - `/video` – Continuous webcam stream with YOLOv5 detections.

- `ai/`
  - `ai_model.py`  
    - `load_yolov5_model()` – Loads YOLOv5 model from `helper/params.py::Parameters.model`.  
    - `detection(frame, model, names)` – Runs YOLOv5 on a frame, returns annotated frame and label.
  - `ocr_model.py`  
    - `easyocr_model_load()` – Initializes EasyOCR reader.  
    - `easyocr_model_works()` – Helper for running OCR on lists of images.  
    - `pytesseract_model_works()` – Alternative OCR using Tesseract.

- `helper/`
  - `general_utils.py`  
    - `filter_text(rectangle_size, ocr_result, region_threshold)` – Filters OCR results by bounding box size.  
    - `save_results(text, csv_filename, folder_path)` – Appends recognized text to a CSV file.
  - `params.py`  
    - `Parameters` class – Central configuration (model path, thresholds, colors, device, etc.).

- `models/`, `utils/`  
  - Core YOLOv5 model definitions and utilities (ported from Ultralytics).

- `model/`
  - `best.pt`, `last.pt` – YOLOv5 weights for plate detection (you must provide/replace these).

- `templates/`
  - `index.html` – Template for the Flask upload page.

- `static/`
  - `image0.jpg` – Last-output image from the Flask upload endpoint.

- `plate_dataset/`  
  Example plate images / dataset (optional).

- `requirements.txt`  
  Python dependencies for the project.

---

## System Flow (Mermaid Diagram)

This diagram shows the **full end-to-end flow** for both modes: **real-time ANPR** (`main.py`) and **web ANPR** (`app.py`).

```mermaid
flowchart TD
    %% ===== ENTRY POINT =====
    A[Start] --> B[Select Mode]
    B --> C[Real-time ANPR (main.py)]
    B --> D[Web ANPR (app.py)]

    %% ===== REAL-TIME PIPELINE (main.py) =====
    subgraph RT[Real-time ANPR - Webcam]
        C --> RT1[Load Parameters\nhelper/params.py -> Parameters]
        RT1 --> RT2[Load YOLOv5 Model\nai/ai_model.load_yolov5_model]
        RT2 --> RT3[Load EasyOCR Reader\naio/ocr_model.easyocr_model_load]
        RT3 --> RT4[Open Webcam\ncv2.VideoCapture(0)]
        RT4 --> RT5[Loop: Capture Frame]

        RT5 --> RT6[Run Detection\nai/ai_model.detection(frame, model, names)]
        RT6 --> RT7[Get Plate Bounding Box\n& Crop Plate Region]
        RT7 --> RT8[Run OCR on Plate\ntext_reader.readtext(cropped_plate)]
        RT8 --> RT9[Filter OCR Results\nhelper/general_utils.filter_text]
        RT9 --> RT10[Append Latest Plate Text\nto ocr_results.csv\n(save_results)]
        RT9 --> RT11[Print Plate Text to Console]

        RT7 --> RT12[Draw Bounding Box & Label\non Frame]
        RT12 --> RT13[Show Frame (cv2.imshow)]
        RT13 --> RT14[Check Key (cv2.waitKey)]
        RT14 --> RT15{ESC Pressed?}
        RT15 -->|No| RT5
        RT15 -->|Yes| RT16[Release Camera & Close Windows]
    end

    %% ===== WEB PIPELINE (app.py) =====
    subgraph WEB[Flask Web ANPR]
        D --> W1[Start Flask App]
        W1 --> W2[Load YOLOv5 Model via torch.hub\npath=model/last.pt]

        %% Upload-based detection (/)
        W2 --> W3[Route '/': Show Upload Form\nindex.html]
        W3 --> W4[User Uploads Image]
        W4 --> W5[Read Image Bytes\nrequest.files['file'].read()]
        W5 --> W6[Convert Bytes to PIL Image\nImage.open(BytesIO)]
        W6 --> W7[Run YOLOv5 on Image\nmodel(img, size=640)]
        W7 --> W8[Render Detections on Image\nresults.render()]
        W8 --> W9[Save Output Image\nstatic/image0.jpg]
        W9 --> W10[Redirect User to static/image0.jpg]

        %% MJPEG streaming (/video)
        W2 --> W11[Route '/video': Start gen()]
        W11 --> W12[Open Webcam\ncv2.VideoCapture(0)]
        W12 --> W13[Loop: Read Frame]
        W13 --> W14[Encode Frame to JPEG\ncv2.imencode]
        W14 --> W15[Wrap as PIL & Run YOLOv5\nmodel(img, size=640)]
        W15 --> W16[Render Detections\nresults.render()]
        W16 --> W17[Convert to BGR & JPEG Bytes]
        W17 --> W18[Yield MJPEG Chunks\nFlask Response]
    end
```

> **Note:** The **OCR step is only in the real-time `main.py` pipeline**. The Flask app, as written, uses YOLOv5 detection and visualizes bounding boxes but does not currently perform OCR in `app.py`.

---

## Installation

### 1. Create and activate a virtual environment (recommended)

From the project root (`Automatic-number-plate-recognition-YOLO-OCR`):

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure YOLOv5 model paths

- In `helper/params.py`, the default is:

  ```python
  self.model = "/home/mef/Documents/plate_detection_project/best.pt"
  ```

  **Change this** to your local weights path, typically:

  ```python
  self.model = "model/best.pt"
  ```

- For `app.py`, YOLOv5 is loaded via:

  ```python
  model = torch.hub.load(
      "ultralytics/yolov5", "custom", path="model/last.pt", force_reload=True
  )
  ```

  Ensure `model/last.pt` exists, or update the `path` argument.

---

## Usage

### A. Real-Time Webcam ANPR (`main.py`)

**Purpose:**  
Run live ANPR from a webcam, perform OCR with EasyOCR, and log results to `ocr_results.csv`.

**Run:**

```bash
python main.py
```

**What it does (step-by-step):**

1. Loads global configuration from `helper/params.py` into a `Parameters` instance.
2. Loads the YOLOv5 plate detection model using `ai/ai_model.load_yolov5_model`.
3. Opens your default webcam (`cv2.VideoCapture(0)`).
4. Starts an infinite loop:
   - Grabs a frame from the webcam.
   - Runs YOLOv5 detection to find license plates.
   - Crops the detected plate region.
   - Runs EasyOCR on the cropped plate.
   - Filters OCR results using size/region threshold (`filter_text`).
   - Saves the latest recognized plate string to `ocr_results.csv` (`save_results`).
   - Overlays bounding boxes and labels on the frame.
   - Displays the annotated frame in a window.
5. Exits when **ESC** is pressed; releases the camera and closes windows.

**Outputs:**

- Console: prints recognized plate text.
- `ocr_results.csv`: contains a list of plate strings (one per line).
- OpenCV window: shows real-time camera feed with bounding boxes and labels.

---

### B. Flask Web App (`app.py`)

**Purpose:**  
Provide a web interface for ANPR-like detection using YOLOv5.

**Run:**

```bash
python app.py
```

By default, the app listens on `http://0.0.0.0:5000`.

#### 1. Upload-based Detection (`/`)

- Go to: `http://localhost:5000`
- You will see an upload form (from `templates/index.html`).
- Steps:
  1. Choose an image containing a vehicle license plate.
  2. Submit the form.
  3. The server:
     - Reads the file.
     - Converts to a PIL image.
     - Runs YOLOv5 detection (`model(img, size=640)`).
     - Calls `results.render()` to draw bounding boxes.
     - Saves the resulting image as `static/image0.jpg`.
     - Redirects you to `static/image0.jpg`, which shows the annotated image.

#### 2. Streaming Detection (`/video`)

- Go to: `http://localhost:5000/video`
- The server:
  - Opens the webcam.
  - In a loop:
    - Reads each frame.
    - Passes it through YOLOv5.
    - Renders detections onto the frame.
    - Encodes the frame as JPEG.
    - Streams it back as an MJPEG stream.

> **Note:** The `/video` route currently **does not do OCR**, just YOLOv5 detection and visualization.

---

## Configuration Details

All important runtime settings are centralized in `helper/params.py`:

- **Model & Hardware**
  - `self.model` – Path to YOLOv5 weights for real-time pipeline.
  - `self.device` – `cuda:0` if available, otherwise `cpu`.

- **Detection & Inference**
  - `self.imgsz` – Input resolution for YOLOv5 (default: 640).
  - `self.conf_thres` – Confidence threshold.
  - `self.max_det` – Max number of detections per image.
  - `self.pred_shape` – Target shape for intermediate image resizing.

- **OCR Filtering**
  - `self.rect_size` – Base rectangle size used for normalization.
  - `self.region_threshold` – Minimum area ratio to accept a text region.

- **Visualization**
  - `self.color_blue`, `self.color_red`, `self.color` – Bounding box colors.
  - `self.rect_thickness` – Thickness of bounding boxes.
  - `self.font_scale`, `self.thickness` – Label font size and thickness.

You can tune these to your camera, environment, and plate characteristics.

---

## Data & Outputs

- **Sample images / datasets**
  - `plate_dataset/` – Plate images (useful for testing OCR functions).
  - `data/images/` – Example images like `bus.jpg`, `zidane.jpg`.

- **Generated outputs**
  - `ocr_results.csv` – Appended with each new recognized plate (real-time pipeline).
  - `static/image0.jpg` – Latest result from the Flask upload endpoint.

---

## Troubleshooting

- **Mermaid diagram not rendering:**
  - Ensure you view the README in a viewer that **supports Mermaid**, such as:
    - GitHub, GitLab, or modern Markdown preview plugins.
  - Make sure the code fence is exactly:
    - Opening fence: ```mermaid  
    - Closing fence: ``` (no extra spaces or indentation).

- **CUDA / GPU issues:**
  - If you don’t have a GPU or CUDA, the code will fall back to CPU:
    ```python
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
  - CPU will be slower; you can reduce `imgsz` or use smaller YOLOv5 weights.

- **Model not found:**
  - Double-check `self.model` in `helper/params.py`.
  - Ensure `model/best.pt` or `model/last.pt` exist and the paths are correct.

---

## License

This project includes YOLOv5-based components and utilities derived from Ultralytics’ YOLOv5.  
See the `LICENSE` file for full license terms and ensure compliance, especially if you plan to use the project commercially.

