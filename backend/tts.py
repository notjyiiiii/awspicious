import torch
from TTS.api import TTS
from torch.serialization import safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add required configs to safe globals for PyTorch 2.6+ compatibility
with safe_globals([XttsConfig, XttsAudioConfig]):
    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Run TTS
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")