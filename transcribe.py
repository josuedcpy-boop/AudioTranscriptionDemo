# This script uses the Whisper model from OpenAI for transcription.
#
# Supported inputs: audio files (m4a, mp3, wav, ...) and video files (mp4, mov, mkv, ...).
# Anything that is not already a .wav is first converted to .wav with pydub (FFmpeg),
# so you can listen to the extracted audio. For video, only the audio track is kept.
#
# Usage:
#   python transcribe.py                 -> transcribes Test.m4a
#   python transcribe.py song.mp3        -> writes song.wav and song.txt
#   python transcribe.py clip.mp4 out.txt
import sys
from pathlib import Path

import whisper
from pydub import AudioSegment

SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}

# Load the Whisper model (use 'base' for faster but less accurate, 'small' or 'medium' for better accuracy)
model = whisper.load_model("base")

# Format seconds as HH:MM:SS.ss (e.g. 75.5 -> 00:01:15.50)
def format_timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"

# Function to transcribe audio (or the audio track of a video)
def transcribe_audio(input_file, output_file):
    input_path = Path(input_file)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Warning: '{input_path.suffix}' is not in the tested list; trying anyway.")

    # If the file is not a wav, convert it to wav (for video, this extracts the audio track)
    if input_path.suffix.lower() != ".wav":
        audio = AudioSegment.from_file(input_path)
        wav_path = input_path.with_suffix(".wav")
        audio.export(wav_path, format="wav")
        print(f"Converted audio saved to {wav_path}")
        input_path = wav_path  # Use the converted wav file

    # Transcribe the audio (FP16 only works on a GPU; this avoids the warning on CPU)
    result = model.transcribe(str(input_path), fp16=(model.device.type == "cuda"))

    # Write the transcription to the output text file, one segment per line:
    # [start --> end]  text
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in result['segments']:
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"[{start} --> {end}]  {segment['text'].strip()}\n")

    print(f"Transcription saved to {output_file}")

# Usage
input_audio = sys.argv[1] if len(sys.argv) > 1 else "Test.m4a"
output_text = sys.argv[2] if len(sys.argv) > 2 else str(Path(input_audio).with_suffix(".txt"))
transcribe_audio(input_audio, output_text)
