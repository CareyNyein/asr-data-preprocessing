import os
import webrtcvad
import pydub
from pydub.utils import make_chunks
from tqdm import tqdm
from pathlib import Path
import csv 

# --- CONFIGURATION ---
INPUT_DIR = "data/to_be_segmented" 
OUTPUT_DIR = "data/segmented_data" 
MAX_SEGMENT_LENGTH_S = 20 
VAD_AGGRESSIVENESS = 3

# --- Audio Loading and Formatting ---
def read_wave(path):
    """
    Reads an audio file, converts it to the required format for webrtcvad,
    and returns the AudioSegment.
    
    VAD requires 16kHz sample rate, 1 channel (mono), and 16-bit depth.
    """
    try:
        audio = pydub.AudioSegment.from_file(path)
    except pydub.exceptions.CouldntDecodeError:
        print(f"Error: Couldn't decode audio file {path}. Check its format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {path}: {e}")
        return None

    # Convert to required format
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio

# --- Voice Activity Detection (VAD) and Segmentation ---
def vad_segment_audio(audio, aggressiveness=1, max_segment_length_s=20):
    """
    Performs VAD on the audio and returns a list of speech segments with
    start and end times in milliseconds.
    """
    vad = webrtcvad.Vad(aggressiveness)

    frame_duration_ms = 30
    frame_size_bytes = int(audio.frame_rate * frame_duration_ms / 1000) * audio.sample_width

    raw_audio_data = audio.raw_data
    frames = []

    # Manually create frames of the correct size for VAD
    for i in range(0, len(raw_audio_data), frame_size_bytes):
        frame = raw_audio_data[i:i + frame_size_bytes]
        if len(frame) == frame_size_bytes:
            frames.append(frame)

    speech_segments = []
    current_segment_start_ms = None

    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, audio.frame_rate)
        
        if is_speech:
            if current_segment_start_ms is None:
                current_segment_start_ms = i * frame_duration_ms
        else:
            if current_segment_start_ms is not None:
                segment_end_ms = i * frame_duration_ms
                speech_segments.append((current_segment_start_ms, segment_end_ms))
                current_segment_start_ms = None

    if current_segment_start_ms is not None:
        speech_segments.append((current_segment_start_ms, len(audio)))
        
    final_segments = []
    if not speech_segments:
        return []

    # --- Segment Merging and Splitting Logic ---
    current_start = speech_segments[0][0]
    current_end = speech_segments[0][1]

    for start, end in speech_segments[1:]:
        if start - current_end < 500: # Merge segments with a small pause (<0.5s)
            current_end = end
        else:
            final_segments.append((current_start, current_end))
            current_start = start
            current_end = end
    final_segments.append((current_start, current_end))
    
    final_final_segments = []
    for start, end in final_segments:
        if end <= start: # Skip any segment with zero or negative duration
            continue
        
        # Split segments longer than the max length
        if (end - start) / 1000 > max_segment_length_s:
            num_splits = int((end - start) / (max_segment_length_s * 1000)) + 1
            segment_length = (end - start) / num_splits
            for i in range(num_splits):
                final_final_segments.append((start + i * segment_length, start + (i + 1) * segment_length))
        else:
            final_final_segments.append((start, end))

    return final_final_segments

# --- Text Chunking (Language Agnostic) ---
def chunk_text(text, max_length=150):
    """
    Splits text into chunks of a given max character length without using
    language-specific tokenizers.
    """
    chunks = []
    text_to_process = text.strip()
    
    while len(text_to_process) > 0:
        if len(text_to_process) <= max_length:
            chunks.append(text_to_process)
            break
        
        split_point = text_to_process.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        
        chunks.append(text_to_process[:split_point])
        text_to_process = text_to_process[split_point:].strip()
        
    return chunks

# --- Main Processing Loop ---
def process_data(input_dir, output_dir):
    """
    Main function to process long audio files and their transcripts.
    Saves audio/text pairs and a summary CSV file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Locate the Transcripts 
    transcript_input_dir = Path(input_dir) 
    
    audio_files = [f for f in os.listdir(input_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
    total_files = len(audio_files)
    
    # --- NEW: Read existing CSV data for appending ---
    csv_file_path = output_path / "segmented_data_catalog.csv"
    
    # Read existing data if the file exists
    if csv_file_path.exists():
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            csv_data = list(reader)
        if not csv_data or csv_data[0] != ['audio_id', 'text']:
             csv_data = [['audio_id', 'text']]
    else:
        csv_data = [['audio_id', 'text']]
        
    total_chunks_saved = 0

    if total_files == 0:
        print(f"Error: No audio files found in {input_path}.")
        return
        
    print(f"Starting segmentation of {total_files} files from '{input_path}'...")
    
    for filename in tqdm(audio_files, desc="Segmenting Audio and Text"):
        base_name = os.path.splitext(filename)[0]
        audio_path = input_path / filename
        
        # Look for the original transcript in the INPUT directory
        transcript_path = transcript_input_dir / f"{base_name}.txt"

        if not transcript_path.exists():
            print(f"\nSkipping {filename}: Original transcript not found at {transcript_path}.")
            continue

        audio = read_wave(str(audio_path))
        if audio is None:
            continue
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            full_transcript = f.read().replace('\n', ' ').strip()
        
        # --- EXECUTE CORE LOGIC ---
        transcript_chunks = chunk_text(full_transcript)
        audio_segments = vad_segment_audio(audio, aggressiveness=VAD_AGGRESSIVENESS, max_segment_length_s=MAX_SEGMENT_LENGTH_S)
        
        # Pair audio segments with transcript chunks
        num_chunks_to_create = min(len(audio_segments), len(transcript_chunks))
        
        if num_chunks_to_create == 0:
            continue

        for i in range(num_chunks_to_create):
            start_ms, end_ms = audio_segments[i]
            transcript_chunk = transcript_chunks[i]
            
            duration_ms = end_ms - start_ms
            if duration_ms <= 0:
                continue
            
            output_audio_name = f"{base_name}_chunk_{i}.wav"
            output_audio_path = output_path / output_audio_name
            
            # --- NEW: File Existence Check (Skip if already processed) ---
            if output_audio_path.exists():
                continue 
            
            # --- Saving Individual Files (Only runs if file does not exist) ---
            
            # Save Audio and Text
            audio_chunk = audio[start_ms:end_ms]
            audio_chunk.export(str(output_audio_path), format="wav")
            
            output_transcript_name = f"{base_name}_chunk_{i}.txt"
            output_transcript_path = output_path / output_transcript_name
            with open(output_transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_chunk)
            
            # --- Recording Data for CSV ---
            csv_data.append([output_audio_name, transcript_chunk])
            total_chunks_saved += 1
                
    # --- FINAL STEP: Write CSV File (rewriting the full list with new appended data) ---
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
                
    print(f"\n✅ Segmentation complete. Total {total_chunks_saved} new chunks appended.")
    print(f"📄 Catalog CSV updated: {csv_file_path.name}")


if __name__ == "__main__":
    process_data(INPUT_DIR, OUTPUT_DIR)
