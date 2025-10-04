
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
# Activate your environment (e.g., conda activate your_env)

# Install core libraries
pip install torch torchaudio tqdm pathlib requests python-dotenv
# Install the Denoiser library (Crucial dependency for enhancement)
pip install denoiser

```

### 2. Project Structure

Organize your raw data into the specified directories:

```
project-sgk/
├── .env                  # Environment variables for download script
├── .gitignore            # Git exclusion file (NEW)
├── noise_reducer.py      # The primary batch processing script.
├── download_audio.py     # New script for downloading raw audio files.
└── data/
    ├── raw_noise_unprocessed/  <-- INPUT: Place all your long, noisy audio files here.
    └── raw_noise_reduced/      <-- OUTPUT: Cleaned audio files will be saved here.

```

### 3. Git Exclusion (`.gitignore`)

It is vital to exclude large model files and caches from Git history. Create a file named `.gitignore` in the project root and add the following contents:

```
# --- Virtual Environments ---
venv/
.venv/
# Add your Conda environment name here if applicable (e.g., asr_env/)

# --- Data and Logs ---
data/
logs/
__pycache__/

# --- PyTorch & Hugging Face Caches ---
# PyTorch Hub cache (where Denoiser model weights are downloaded)
.cache/
.pt_ext/
.ipynb_checkpoints/
*.pth
*.pt

# --- MacOS/Windows Specific Files ---
.DS_Store
Thumbs.db

```

## 🚀 Usage Guide

### 0. Data Acquisition (Download)

_This step is optional if you have you own audio files. The provided script is an example tailored to specific data needs and may require modification._

**Requirements:** Create a file named `.env` in the root of your project and set the base URL for your audio files:

```
# .env file content
AUDIO_BASE_URL="[http://your-remote-server.com/audio/daily_broadcast](http://your-remote-server.com/audio/daily_broadcast)_{date}.mp3"

```

**Execution:**

```
python download_audio.py

```

### Batch Enhancement (Command Line Interface)

Use this mode to process all files in your input folder (`data/raw_noise_unprocessed`) at once.

#### **Control Noise Reduction Aggressiveness**

The reduction level is controlled by the `NOISE_GAIN_DB` variable inside `noise_reducer.py`.

-   **Clarification for Your Use Case**: The default setting (`NOISE_GAIN_DB = 0`) is suitable for general use, and in your particular case of removing background music, you may not need to tweak it.
    
-   **Tweakable Parameter**: If someone else needs more or less aggressive noise reduction, they can tweak the `NOISE_GAIN_DB` variable inside `noise_reducer.py`.
    
    -   **Positive Gain (e.g., `5`):** Less aggressive removal; preserves more of the ambient noise and speech quality.
        
    -   **Zero Gain (`0`, Default):** Standard noise suppression, best for general use.
        
    -   **Negative Gain (e.g., `-10`):** Highly aggressive removal; suppresses more background noise but risks degrading speech quality.
        

**To adjust the gain, modify the `NOISE_GAIN_DB` variable inside the script.**

```
# Run the batch enhancement script:
python noise_reducer.py

```

The script will display a progress bar, skip already processed files, and save the clean output to `data/raw_noise_reduced`.

## ✅ Next Steps (ASR Data Preparation)

Once you have your clean audio files in the output directory, your next step is to perform segmentation and create the final dataset for Whisper fine-tuning:

1.  **Segmentation**: Run your separate **VAD-based segmentation script**, ensuring the input directory is set to `data/raw_noise_reduced`. This creates the short, clean audio/transcript pairs.
    
2.  **Dataset Creation**: Use the generated segmented files to construct the Hugging Face `Dataset` object, which is the required format for starting the Whisper training process.