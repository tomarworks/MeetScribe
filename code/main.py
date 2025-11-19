import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import webrtcvad
import queue
import time
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import subprocess
import json
import uuid

# DYNAMIC NAMING
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "audio_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"recording_{timestamp}"
AUDIO_FILE = os.path.join(UPLOAD_DIR, f"{base_filename}.wav")
TRANSCRIPT_TEXT = os.path.join(UPLOAD_DIR, f"{base_filename}.txt")

#  AUDIO RECORDING CONFIG
fs = 16000
frame_duration = 30
frame_size = int(fs * frame_duration / 1000)
channels = 1
vad = webrtcvad.Vad(2)

audio_queue = queue.Queue()
recorded_frames = []
silence_duration_limit = 2.0
silence_start_time = None
recording = True

print("\U0001F399\ufe0f Speak now. Recording will stop after 1 sec of silence...")

def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, fs)

def callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', blocksize=frame_size, callback=callback):
    while recording:
        try:
            data = audio_queue.get(timeout=0.1)
            frame_bytes = data.tobytes()

            if is_speech(frame_bytes):
                silence_start_time = None
                recorded_frames.append(data)
                print("\U0001F5E3\ufe0f Speaking...")
            else:
                print("\U0001F507 Silence")
                recorded_frames.append(data)
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > silence_duration_limit:
                    print("\u23f9\ufe0f No speech for 1 second. Stopping...")
                    recording = False
        except queue.Empty:
            pass

# SAVE AUDIO 
final_audio = np.concatenate(recorded_frames, axis=0)
wav.write(AUDIO_FILE, fs, final_audio)
print(f"\U0001F4C1 Audio saved as {AUDIO_FILE}")

# ==== TRANSCRIBE ====
print("\U0001F524 Transcribing using faster-whisper...")
model = WhisperModel("medium", compute_type="int8")
segments, _ = model.transcribe(AUDIO_FILE, word_timestamps=True, vad_filter=True)
transcript_segments = list(segments)

# ==== DIARIZATION ====
# ==== DIARIZATION ====
print("\U0001F9E0 Performing speaker diarization...")
HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxxxxxx"  # your token
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
diarization = pipeline(AUDIO_FILE)

print("\U0001F9EC Computing speaker embeddings and merging similar speakers...")
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cpu")
audio = Audio()
speaker_embeddings = {}
merged_mapping = {}
speaker_counter = 1

# Higher threshold to avoid merging different people
MERGE_THRESHOLD = 0.88  

for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker in speaker_embeddings:
        continue

    waveform, sample_rate = audio.crop(AUDIO_FILE, turn)
    embedding = embedding_model(waveform[None])[0]

    matched = False
    for existing_speaker, existing_embedding in speaker_embeddings.items():
        sim = cosine_similarity([embedding], [existing_embedding])[0][0]
        if sim >= MERGE_THRESHOLD:
            merged_mapping[speaker] = merged_mapping[existing_speaker]
            matched = True
            break

    if not matched:
        new_label = f"speaker{speaker_counter}"
        merged_mapping[speaker] = new_label
        speaker_embeddings[speaker] = embedding
        speaker_counter += 1

# Map diarization output with merged speaker labels
speaker_segments = [
    {"start": turn.start, "end": turn.end, "speaker": merged_mapping[speaker]}
    for turn, _, speaker in diarization.itertracks(yield_label=True)
]

print("\U0001F4DD Matching transcripts to speakers...")
final_output = []
for segment in transcript_segments:
    seg_start, seg_end, seg_text = segment.start, segment.end, segment.text
    assigned_speaker = "Unknown"
    for s in speaker_segments:
        if not (seg_end < s["start"] or seg_start > s["end"]):
            assigned_speaker = s["speaker"]
            break
    final_output.append({
        "speaker": assigned_speaker,
        "start": seg_start,
        "end": seg_end,
        "text": seg_text.strip()
    })

# Save text transcript
with open(TRANSCRIPT_TEXT, "w") as f:
    for line in final_output:
        f.write(f"{line['speaker']}: {line['text']}\n")

# Save diarized JSON for UI clickable playback
import json
JSON_TRANSCRIPT = os.path.join(UPLOAD_DIR, f"{base_filename}.json")
with open(JSON_TRANSCRIPT, "w") as jf:
    json.dump(final_output, jf, indent=2)

print(f"✅ Saved diarized transcript: {TRANSCRIPT_TEXT}")
print(f"✅ Saved JSON for UI: {JSON_TRANSCRIPT}")

# Save raw transcript
with open(TRANSCRIPT_TEXT, "w") as f:
    for line in final_output:
        f.write(f"{line['speaker']}: {line['text']}\n")
print("\U0001F4DD transcript.txt saved for UI generation.")

print("🧩 Building word-level data aligned to speakers...")
word_level_data = []
for segment in transcript_segments:
    if getattr(segment, "words", None):
        for w in segment.words:
            assigned_speaker = "Unknown"
            for s in speaker_segments:
                # overlap check between word and diarization turn
                if not (w.end < s["start"] or w.start > s["end"]):
                    assigned_speaker = s["speaker"]
                    break
            word_level_data.append({
                "word": w.word,
                "start": float(w.start) if w.start is not None else None,
                "end": float(w.end) if w.end is not None else None,
                "speaker": assigned_speaker
            })

# save word-level JSON for the UI in the SAME folder as the audio/text
JSON_WORDS = os.path.join(UPLOAD_DIR, f"{base_filename}_words.json")
with open(JSON_WORDS, "w", encoding="utf-8") as f:
    json.dump(word_level_data, f, ensure_ascii=False, indent=2)
print(f"✅ Saved word-level JSON for UI: {JSON_WORDS}")


# ==== SAVE MEETING ENTRY (with UUID) ====
meeting_id = str(uuid.uuid4())
meeting_entry = {
    "id": meeting_id,   # stays fixed forever
    "name": os.path.splitext(os.path.basename(AUDIO_FILE))[0],  # user-facing name
    "audio": os.path.basename(AUDIO_FILE),
    "transcript": os.path.basename(JSON_TRANSCRIPT),
    "words": os.path.basename(JSON_WORDS),
    "created": datetime.now().isoformat()
}

MEETINGS_FILE = os.path.join(UPLOAD_DIR, "meetings.json")
if os.path.exists(MEETINGS_FILE):
    with open(MEETINGS_FILE, "r") as f:
        meetings = json.load(f)
else:
    meetings = []

meetings.append(meeting_entry)

with open(MEETINGS_FILE, "w") as f:
    json.dump(meetings, f, indent=2)

print(f"📁 Updated meetings.json with ID {meeting_id}")


# ==== LAUNCH UI WITH AUDIO + TRANSCRIPT ====
print("\U0001F310 Launching HTML UI...")
script_path = os.path.join(os.path.dirname(__file__), "new_UI.py")
subprocess.run(["python", script_path, AUDIO_FILE, TRANSCRIPT_TEXT, JSON_WORDS])
