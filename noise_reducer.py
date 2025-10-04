import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import glob
from pathlib import Path
import torchaudio
from tqdm import tqdm
import torch
import torch.nn.functional as F 

# --- CONFIGURATION ---
INPUT_DIR = "data/raw_noise_unprocessed" 
OUTPUT_DIR = "data/raw_noise_reduced" 

# Use GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- MODEL LOADING ---
enhance_model = None
try:
    model_callable = 'dns64'
    enhance_model = torch.hub.load(
        'facebookresearch/denoiser', 
        model_callable, 
        pretrained=True
    ).to(DEVICE)
    
except Exception as e:
    print(f"Error loading Denoiser model: {e}")
    print("WARNING: Cannot load Denoiser model. Please ensure you have run:")
    print("         pip install denoiser")
    print("         and check your internet connection.")
    # Set enhance_model to None if loading fails
    enhance_model = None


def clean_audio_file(input_path: Path, output_path: Path):
    """
    Loads an audio file, performs speech enhancement, and saves the cleaned result.
    This function now uses the structure required by the Facebook Denoiser.
    """
    if enhance_model is None:
        return 
    
    # Target chunk size
    MAX_CHUNK_SECONDS = 10 
    
    try:
        # --- AUDIO LOADING: Direct torchaudio load ---
        waveform, sample_rate = torchaudio.load(str(input_path)) 
        
        # --- Pre-processing/Resampling/Mono Conversion ---
        waveform = waveform.to(DEVICE)
        
        # Denoiser model requires 16kHz audio
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(sample_rate, 16000).to(DEVICE) 
            waveform = transform(waveform)
            sample_rate = 16000
            
        # Ensure mono channel (Denoiser model expects mono)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        
        # --- CHUNKING LOGIC FOR PADDING ---
        
        original_total_samples = waveform.shape[-1] 
        samples_per_chunk = sample_rate * MAX_CHUNK_SECONDS
        num_chunks = (original_total_samples + samples_per_chunk - 1) // samples_per_chunk
        
        enhanced_chunks = []
        
        # Process audio in chunks
        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = min((i + 1) * samples_per_chunk, original_total_samples)
            
            chunk = waveform[:, start_sample:end_sample]
            
            # --- PADDING: Ensure fixed size for model ---
            current_chunk_size = chunk.shape[-1]
            if current_chunk_size < samples_per_chunk:
                padding_needed = samples_per_chunk - current_chunk_size
                chunk = F.pad(chunk, (0, padding_needed), 'constant', 0)
                
            with torch.no_grad():
                
                # --- DENOISER INPUT PREP ---
                chunk_batch = chunk.unsqueeze(0) 

                # Denoiser call
                enhanced_source = enhance_model(chunk_batch)
                # feel free to tweak as follow. 
                # enhanced_source = enhance_model(chunk_batch, noise_gain_db=NOISE_GAIN_DB)
                
                # --- Preparation for Concatenation ---
                enhanced_chunk_padded = enhanced_source.squeeze(0)

                enhanced_chunks.append(enhanced_chunk_padded) 

        # Concatenate all enhanced chunks back
        enhanced_waveform_padded = torch.cat(enhanced_chunks, dim=-1)

        # --- FINAL STEP: Slice off the total padding using the original length ---
        enhanced_waveform = enhanced_waveform_padded[:, :original_total_samples]

        # Save the cleaned audio file
        torchaudio.save(
            str(output_path),
            enhanced_waveform.cpu(), 
            sample_rate,
        )
        
    except RuntimeError as e:
        if 'CUDA' in str(e) and 'memory' in str(e):
             print(f"Error processing {input_path.name}: CUDA memory (OOM) occurred. Original Error: {e}")
        else:
             print(f"Error processing {input_path.name}: {e}")
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")

def main():
    if enhance_model is None:
        print("Cannot proceed without a loaded enhancement model.")
        return

    # Use Path objects for robust directory handling
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # Ensure the input directory exists
    if not input_path.is_dir():
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please ensure your long audio files are in this folder.")
        return

    # Create the output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Define audio file extensions to search for
    extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac']
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.glob(ext))
        
    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}. Please check the directory and file extensions.")
        return

    print(f"Found {len(audio_files)} audio files for enhancement. Starting process...")
    
    # Process files with a progress bar (tqdm)
    files_to_process = 0
    
    # First pass: Determine how many files need processing
    # In case some of the files have already been processed
    for file_path in audio_files:
        output_file = output_path / file_path.name

        if output_file.exists():
            print(f"Skipping {file_path.name}: Cleaned file already exists in {OUTPUT_DIR}.")
            continue

        files_to_process += 1
        
    if files_to_process == 0:
        print("All files have already been processed and exist in the output directory. Nothing to do.")
        return
        
    # Second pass: Re-iterate and process only the necessary files
    with tqdm(total=files_to_process, desc="Enhancing Audio Files") as pbar:
        for file_path in audio_files:
            output_file = output_path / file_path.name

            if output_file.exists():
                pbar.update(1)
                continue
                
            clean_audio_file(file_path, output_file)
            pbar.update(1)
    
    print("\n--- Audio Enhancement Complete ---")
    print(f"Cleaned files are ready in the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
