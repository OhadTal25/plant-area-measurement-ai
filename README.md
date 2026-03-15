# Plant Area Tool, SAM-based leaf detection and area measurement

This repository contains a Google Colab project that detects plant leaves from an image, segments them with Meta's Segment Anything Model, and estimates plant area in cm² after image calibration.

## What the project does

- Upload a plant image
- Calibrate the image by clicking 2 reference points
- Detect plant regions with SAM
- Filter masks using a green HSV mask
- Measure each detected plant area
- Export:
  - CSV with per-plant measurements
  - PNG overlay image
  - Histogram of area distribution

## Main features

- SAM automatic segmentation
- Green pixel filtering in HSV
- Manual calibration using a known reference length
- Single plant selection
- Multi-select by clicking
- Box selection for grouped measurements
- Gradio UI for interactive analysis

## Project structure

```text
plant-area-tool-repo/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── notebook/
    └── SAM3_NEW.ipynb
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-area-tool.git
cd plant-area-tool
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Download the SAM checkpoint

Download this file and place it in the project root:

```text
sam_vit_h_4b8939.pth
```

Official checkpoint source:
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

## Run locally

```bash
python app.py
```

## Notes

- The model checkpoint is large, so it should not be committed to GitHub.
- If you mainly want to share the code from Colab, you can also keep the notebook version in the repo and add `app.py` as the cleaner runnable version.
- If you want, you can later split the code into:
  - `src/segmentation.py`
  - `src/ui.py`
  - `src/utils.py`

## Suggested GitHub description

SAM-based plant leaf segmentation and area measurement from images, with calibration, interactive selection, CSV export, and Gradio UI.

## Suggested topics

`segment-anything`, `sam`, `computer-vision`, `plant-phenotyping`, `image-segmentation`, `gradio`, `opencv`, `python`
