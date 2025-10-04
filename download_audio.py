import requests
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("AUDIO_BASE_URL")

# Directory to save audio files
save_dir = "data/raw_noise_unprocessed"
os.makedirs(save_dir, exist_ok=True)

# Set your date range
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 10, 31)

current_date = start_date
while current_date <= end_date:
    # Format date as YYYY-MM-DD (with leading zeros)
    date_str = current_date.strftime("%Y-%m-%d")
    file_url = base_url.format(date=date_str)

    # Save under data/raw_noise_unprocessed/
    filename = os.path.join(save_dir, f"audio_{date_str}.mp3")

    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists.")
    else:
        try:
            print(f"⬇️ Downloading {file_url} ...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ Saved {filename}")
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")

    current_date += timedelta(days=1)
