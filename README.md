🎙️ PyTorch Audio Enhancement Pipeline for ASR Data Prep
This project provides a robust solution for preprocessing noisy audio data, specifically designed to prepare high-quality training material for fine-tuning Automatic Speech Recognition (ASR) models like Whisper.

The core pipeline uses an advanced neural network (Facebook Denoiser) to eliminate background noise and music from long audio recordings.

✨ Features
Noise Reduction (Denoiser): Uses the Facebook Denoiser model (via PyTorch Hub) to clean speech signals by removing complex background noise and music.

Batch Processing (CLI): Process hundreds of audio files automatically via the command line, including logic to skip already processed files.

GPU/CUDA Optimization: Optimized for speed on NVIDIA GPUs, including fixes for CUDA Out of Memory (OOM) fragmentation.

Robust Chunking & Padding: Handles extremely long audio files by chunking them into 10-second segments, processing them individually, and reassembling them without errors.

⚙️ Prerequisites & Setup
This project requires a Python environment (preferably Python 3.8+) and FFmpeg installed on your system.

1. Environment and Dependencies
Ensure your Python virtual environment is active, and install all required libraries:

# Activate your environment (e.g., source (your_env/bin/activate))

# Install core libraries
pip install torch torchaudio tqdm pathlib requests python-dotenv
# Install the Denoiser library (Crucial dependency for enhancement)
pip install denoiser

2. Project Structure
Organize your raw data into the specified directories:

project-sgk/
├── noise_reducer.py      # The primary batch processing script.
├── download_audio.py     # New script for downloading raw audio files.
└── data/
    ├── raw_noise_unprocessed/  <-- INPUT: Place all your long, noisy audio files here.
    └── raw_noise_reduced/      <-- OUTPUT: Cleaned audio files will be saved here.

🚀 Usage Guide
0. Data Acquisition (Download)
This step is optional if you already have your audio files. The provided script is an example tailored to specific data needs and may require modification.

Requirements: Create a file named .env in the root of your project and set the base URL for your audio files:

# .env file content
AUDIO_BASE_URL="[http://your-remote-server.com/audio/daily_broadcast](http://your-remote-server.com/audio/daily_broadcast)_{date}.mp3"

Execution:

python download_audio.py

Batch Enhancement (Command Line Interface)
Use this mode to process all files in your input folder (data/raw_noise_unprocessed) at once.

# Run the batch enhancement script:
python noise_reducer.py


The script will display a progress bar, skip already processed files, and save the clean output to data/raw_noise_reduced.