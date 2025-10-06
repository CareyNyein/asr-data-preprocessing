
# 🎙️ PyTorch Audio Enhancement Pipeline for ASR Data Prep

This project provides a robust solution for preprocessing noisy audio data, specifically designed to prepare high-quality training material for fine-tuning Automatic Speech Recognition (ASR) models like Whisper.

The core pipeline uses an advanced neural network (Facebook Denoiser) to eliminate background noise and music from long audio recordings.

## ✨ Features

-   **Noise Reduction (Denoiser)**: Uses the Facebook Denoiser model (via PyTorch Hub) to clean speech signals by removing complex background noise and music.
    
-   **Batch Processing (CLI)**: Process hundreds of audio files automatically via the command line, including logic to skip already processed files.
    
-   **GPU/CUDA Optimization**: Optimized for speed on NVIDIA GPUs, including fixes for CUDA Out of Memory (OOM) fragmentation.
    
-   **Robust Chunking & Padding**: Handles extremely long audio files by chunking them into 10-second segments, processing them individually, and reassembling them without errors.
    

## ⚙️ Prerequisites & Setup

This project requires a Python environment (preferably **Python 3.8+**) and **FFmpeg** installed on your system.

### 1. Environment and Dependencies

Ensure your Python virtual environment is active, and install all required libraries:

```
# Activate your environment (e.g., source your_venv/bin/activate)

# Install core libraries
pip install torch torchaudio tqdm pathlib requests python-dotenv
# Install the Denoiser library (Crucial dependency for enhancement)
pip install denoiser

OR 

pip install -r requirements.txt


```

### 2. Project Structure

Organize your raw data into the specified directories:

```
project-sgk/
├── .env                  # Environment variables for download script
├── .gitignore            # Git exclusion file (NEW)
├── noise_reducer.py      # The primary batch processing script.
├── download_audio.py     # Script for downloading raw audio files.
├── vad_segmentation.py   # NEW: Segmentation and CSV cataloging script.
└── data/
    ├── raw_noise_unprocessed/  <-- Original Audio 
    ├── raw_noise_reduced/      <-- Cleaned Audio Files (Input for Step 2)
    ├── to_be_segmented/        <-- Cleaned Audio Files & Transcripts (Input for Step 2)
    └── segmented_data/         <-- Output: Final Chunks & Catalog CSV


```

## 🚀 Usage Guide

### 0. Data Acquisition (Download)

_This step is optional if you have you own audio files. The provided script is an example tailored to specific data needs and may require modification._

**Requirements:** Create a file named `.env` in the root of your project and set the base URL for your audio files:

```
# .env file content
AUDIO_BASE_URL="[http://your-remote-server.com/](http://your-remote-server.com/)_{date}.mp3"


```

**Execution:**

```
python download_audio.py


```

### 1. Audio Enhancement (Noise Reduction)

Use this script to clean all files in your input folder (`data/raw_noise_unprocessed`) and save the output to `data/raw_noise_reduced`.

**Control Noise Reduction Aggressiveness**

The reduction level is controlled by the `NOISE_GAIN_DB` variable inside `noise_reducer.py`.

-   **Tweakable Parameter**: To adjust the gain, modify the `NOISE_GAIN_DB` variable inside the script.
    
    -   **Zero Gain (`0`, Default):** Standard noise suppression, generally good for removing music without degrading speech.
        

```
# Run the batch enhancement script:
python noise_reducer.py


```

### 2. ASR Data Segmentation and Cataloging

This script processes the **cleaned audio files**, performs VAD to find speech segments, pairs them with text chunks, and generates the final files ready for manual correction and model training.

**Input Directory:**  `data/raw_noise_reduced/`  **Output Directory:**  `data/segmented_data/`

**Output Files:** The script creates individual `.wav` and `.txt` files, and a crucial master catalog: `data/segmented_data/segmented_data_catalog.csv`. This CSV contains `[audio_id, text]` pairs, which you will use for manual review and correction before fine-tuning Whisper.

```
# Run the segmentation script:
python vad_segmentation.py

```