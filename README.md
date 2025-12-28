This is a complete, start-to-finish guide to cloning the "Angel" voice from GTA V for use in Home Assistant. This guide consolidates every command you need: downloading, cleaning, cutting, transcribing, and training.

### **Prerequisites**

You need  Ubuntu22.04  (or WSL2 on Windows) with an NVIDIA GPU and `Docker` installed.

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
yt-dlp -x --audio-format wav --output "angel_source.wav" "https://www.youtube.com/watch?v=dCGTvZ9ob70"

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


### **Step 4: Install Piper for Training**

1. Create training folders and clone Piper.
```bash
mkdir training
cd training
git clone https://github.com/rhasspy/piper.git
python3 -m venv .venv
source .venv/bin/activate

```


2. **Crucial:** Install specific dependency versions to avoid errors.
```bash
python3 -m pip install pip==23.3.1
pip install numpy==1.24.4
pip install torchmetrics==0.11.4

```


3. Build Piper requirements.
```bash
cd piper/src/python
python3 -m pip install --upgrade wheel setuptools
pip3 install -e .
./build_monotonic_align.sh

```



---

### **Step 4: Pre-process and Train**

1. **Pre-process the data** (Prepare it for the AI).
```bash
python3 -m piper_train.preprocess \
  --language en \
  --input-dir ~/angel_voice_project/raw/ \
  --output-dir ~/train-me \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050

```


2. **Download a base model** (Don't start from scratch).
```bash
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/amy/medium/epoch%3D6679-step%3D1554200.ckpt

```


3. **Start Training.**
*Note: Update the paths to match where your data and the `.ckpt` file you just downloaded are located.*
```bash
python3 -m piper_train \
    --dataset-dir ~/train-me \
    --accelerator 'gpu' \
    --gpus 1 \
    --batch-size 32 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint "/path/to/downloaded/checkpoint.ckpt" \
    --checkpoint-epochs 1 \
    --precision 16 \
    --max-phoneme-ids 400 \
    --quality medium

```


*Press `Ctrl+C` when you are satisfied with the training logic (or when loss is low).*

---

4. **Resume Training.**

Create a file `resume.sh` in the `training/piper/src/python` folder

```bash
#!/bin/bash

# 1. Define the directory where checkpoints are located
CKPT_DIR="~/"

# 2. Find the file with the highest epoch/step (ignoring Zone.Identifier files)
# We search specifically inside CKPT_DIR
LATEST_CKPT=$(ls "$CKPT_DIR"/epoch=*-step=*.ckpt 2>/dev/null | grep -v "Zone.Identifier" | sort -V | tail -n 1)

# 3. Check if we actually found a file
if [ -z "$LATEST_CKPT" ]; then
    echo "❌ Error: No checkpoint files found in $CKPT_DIR!"
    exit 1
fi

echo "✅ Resuming from latest checkpoint: $LATEST_CKPT"

# 4. Run the training command
python3 -m piper_train \
    --dataset-dir ~/train-me \
    --accelerator 'gpu' \
    --gpus 1 \
    --batch-size 16 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint "$LATEST_CKPT" \
    --checkpoint-epochs 1 \
    --precision 16 \
    --max-phoneme-ids 400 \
    --quality medium

```
 **Run the  resume script.**
```bash
chmod +x resume.sh
./resume.sh

```


### **5. Auto-Export Script**

Save this as `export_model.sh`. This script finds the latest checkpoint, converts it to an ONNX file, and automatically sets up the required JSON config file for you.
change the export path
`OUTPUT_DIR="$HOME/train-me/output"`
and the  Name of your voice model
`VOICE_NAME="my_custom_voice"`
```bash
#!/bin/bash

# --- CONFIGURATION ---
# We use $HOME instead of ~ to ensure it finds the path correctly
LOGS_DIR="$HOME/train-me/lightning_logs"

# Where you want the final model saved (Change this if needed)
OUTPUT_DIR="$HOME/train-me/output"

# Name of your voice model
VOICE_NAME="my_custom_voice"

# Path to your config.json (usually in your dataset root)
# Update this path if your config.json is elsewhere!
CONFIG_PATH="$HOME/train-me/config.json"
# ---------------------

# 1. Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# 2. Find the latest version directory
LATEST_VERSION_DIR=$(ls -d "$LOGS_DIR"/version_* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_VERSION_DIR" ]; then
    echo "❌ Error: No version directories found in $LOGS_DIR"
    echo "   (Make sure the path is correct and not using '~' inside quotes)"
    exit 1
fi

# 3. Find the latest checkpoint file
LATEST_CKPT=$(ls "$LATEST_VERSION_DIR/checkpoints"/epoch=*-step=*.ckpt 2>/dev/null | grep -v "Zone.Identifier" | sort -V | tail -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ Error: No checkpoint files found in $LATEST_VERSION_DIR/checkpoints"
    exit 1
fi

echo "✅ Exporting checkpoint: $LATEST_CKPT"

# 4. Export to ONNX
python3 -m piper_train.export_onnx \
    "$LATEST_CKPT" \
    "$OUTPUT_DIR/$VOICE_NAME.onnx"

# 5. Copy and rename the config file
if [ -f "$CONFIG_PATH" ]; then
    cp "$CONFIG_PATH" "$OUTPUT_DIR/$VOICE_NAME.onnx.json"
    echo "✅ Config file copied to $OUTPUT_DIR/$VOICE_NAME.onnx.json"
else
    echo "⚠️ Warning: config.json not found at $CONFIG_PATH."
    echo "   Please manually copy your config.json to $OUTPUT_DIR/$VOICE_NAME.onnx.json"
fi

echo "🎉 Export Complete! Model saved to $OUTPUT_DIR"

```

 **Run the script:**
```bash
chmod +x  export_model.sh
./export_model.sh
```






