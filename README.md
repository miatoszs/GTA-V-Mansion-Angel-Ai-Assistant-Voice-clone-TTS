This is a complete, start-to-finish guide to cloning the "Angel" voice from GTA V for use in Home Assistant. This guide consolidates every command you need: downloading, cleaning, cutting, transcribing, and training.

### **Prerequisites**

You need a Linux environment (or WSL2 on Windows) with an NVIDIA GPU and `Docker` installed.

Install the required system tools first:

```bash
# Update and install ffmpeg, python, and pip
sudo apt update && sudo apt install ffmpeg python3 python3-pip git -y

# Install Python libraries for downloading and transcribing
pip install yt-dlp openai-whisper

```

---

### **Phase 1: Get the Source Audio**

First, we create a workspace and download the high-quality audio source.

**1. Create a workspace folder**

```bash
mkdir -p ~/angel_voice_project/raw
cd ~/angel_voice_project/raw

```

**2. Download the audio from YouTube**
Replace the URL below with the best "Angel Voice Lines" video you found (look for "no music" or "voice lines only").

```bash
yt-dlp -x --audio-format wav --output "angel_source.wav" "https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE"

```

---

### **Phase 2: Audio Cleanup & Segmentation**

We need to remove silence and chop the long file into small 10–15 second chunks for the AI to learn from.

**1. Remove Silence**
This command removes long pauses so the AI doesn't learn to be silent.

```bash
ffmpeg -i angel_source.wav -af "silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB" angel_nosilence.wav

```

**2. Cut the file into segments**
We will slice the clean audio into 15-second chunks. Piper trains best on clips between 5–15 seconds.

```bash
# Create a directory for the clips
mkdir -p wav

# Split the file
ffmpeg -i angel_nosilence.wav -f segment -segment_time 15 -c copy "./wav/split_%03d.wav"

```

**3. Format Correction (Critical Step)**
Piper *requires* specific audio settings (22050Hz, Mono, 16-bit). If you skip this, training will fail.

```bash
mkdir -p wav_final
for file in ./wav/*.wav; do
    ffmpeg -i "$file" -ac 1 -ar 22050 -sample_fmt s16 "./wav_final/$(basename "$file")"
done

# Cleanup: remove the temporary folder and use the final one
rm -rf wav
mv wav_final wav

```

---

### **Phase 3: Automated Transcription**

Now we match the audio clips to text using your Whisper script.

**1. Create the Python script**
Create a file named `transcribe.py`:

```bash
nano transcribe.py

```

**2. Paste your code**
(I have optimized it slightly to handle errors and ensure correct encoding).

```python
import os
import whisper

# Load the model (using 'medium' is better for accuracy than 'base')
print("Loading Whisper model...")
model = whisper.load_model("medium")

audio_dir = "./wav"
output_csv = "./metadata.csv"

# Get files
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
audio_files.sort()

print(f"Found {len(audio_files)} files. Starting transcription...")

with open(output_csv, "w", encoding="utf-8") as f:
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Transcribe
        result = model.transcribe(audio_path)
        transcription = result["text"].strip()
        
        # Simple cleanup to remove hallucinations (optional but recommended)
        if not transcription:
            continue

        file_id = os.path.splitext(audio_file)[0]
        # Format: filename|text
        f.write(f"{file_id}|{transcription}\n")
        print(f"Transcribed {file_id}: {transcription}")

print(f"Done! Metadata saved to {output_csv}")

```

*Press `Ctrl+X`, then `Y`, then `Enter` to save.*

**3. Run the script**

```bash
python3 transcribe.py

```

You now have a `wav` folder and a `metadata.csv` file. You are ready to train.

---

### **Phase 4: Training with TextyMcSpeechy**

We will use the Dockerized tool to train the model locally.

**1. Clone the Training Tool**
Go back to your home directory and get TextyMcSpeechy.

```bash
cd ~
git clone https://github.com/domesticatedviking/TextyMcSpeechy.git
cd TextyMcSpeechy

```

**2. Start the Docker Container**

```bash
./run_container.sh

```

**3. Create the "Dojo" (Project Folder)**

```bash
./create_dojo.sh Angel

```

**4. Move your data into the Dojo**
Now we take the files from Phase 2 & 3 and put them where the trainer can see them.

```bash
# Copy wavs
cp -r ~/angel_voice_project/raw/wav/* ./Angel_dojo/training_folder/wavs/

# Copy metadata
cp ~/angel_voice_project/raw/metadata.csv ./Angel_dojo/training_folder/

```

**5. Download a Base Voice (Fine-Tuning)**
We don't train from scratch; we "fine-tune" an existing voice.

```bash
./download_defaults.sh

```

* **Prompt:** Select an English female voice (e.g., `en_US-amy-medium`).

---

### **Phase 5: Running the Training**

**1. Start the Training**

```bash
./run_training.sh Angel

```

**2. What to do now?**

* **Wait:** A terminal dashboard will appear. Watch the "Loss" graph; it should go down.
* **Listen:** Check the `Angel_dojo/training_folder/output/` folder periodically. The tool will generate test audio there.
* **Stop:** When the voice sounds like Angel (usually 2–4 hours or ~3000 epochs), press `Ctrl+C`.

---

### **Phase 6: Export & Install**

After you stop the training (`Ctrl+C`), the tool automatically exports the finished model.

**1. Locate the files**

```bash
ls -l Angel_dojo/tts_voices/

```

You will see `en_US-angel-medium.onnx` and `en_US-angel-medium.onnx.json`.

**2. Copy to Home Assistant**
(Assuming you have Samba set up, or use `scp`):

```bash
# Example scp command if you have SSH access to Home Assistant
scp Angel_dojo/tts_voices/en_US-angel-medium.onnx* root@homeassistant.local:/share/piper/

```

**3. Activate**

1. Restart the **Piper** add-on in Home Assistant.
2. Go to **Settings > Voice Assistants > Assist**.
3. Set TTS to **Piper** and Voice to **angel (medium)**.

### **Summary of Commands (Quick Reference)**

```bash
# 1. Download
yt-dlp -x --audio-format wav -o "source.wav" "URL"

# 2. Clean & Segment
ffmpeg -i source.wav -af "silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB" clean.wav
ffmpeg -i clean.wav -f segment -segment_time 15 -c copy "split_%03d.wav"

# 3. Transcribe
python3 transcribe.py

# 4. Train
cd TextyMcSpeechy
./run_container.sh
./create_dojo.sh Angel
# (Move files to Angel_dojo/training_folder/)
./download_defaults.sh
./run_training.sh Angel

```