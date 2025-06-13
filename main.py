import numpy as np
import soundfile as sf
from transformers import pipeline
from TTS.utils.synthesizer import Synthesizer
import collections
from TTS.utils import radam
import torch
from params import * 

def tts_vietnamese(text, output_path):
    print("Generating Vietnamese...")
    tts_pipeline = pipeline("text-to-speech", model="models/mms-tts-vie")
    output = tts_pipeline(text)
    audio = np.array(output["audio"], dtype=np.float32).squeeze()
    sf.write(output_path, audio, output["sampling_rate"], subtype='PCM_16')
    print(f"Saved: {output_path}")
    
# Set paths
def tts_japanese(text, output_path):
    safe_globals = [radam.RAdam, collections.defaultdict, dict]
    with torch.serialization.safe_globals(safe_globals):
        synthesizer = Synthesizer(
            tts_checkpoint=TTS_JA_MODEL_PATH,
            tts_config_path=TTS_JA_CONFIG_PATH,
            vocoder_checkpoint=TTS_JA_VOCODER_PATH,
            vocoder_config=TTS_JA_VOCODER_CONFIG_PATH,
            use_cuda=TTS_JA_USE_CUDA
        )

    # Synthesize
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_path)

def main(): 
    vi_path = 'data/data_vi.txt'
    with open(vi_path, 'r', encoding='utf-8') as f:
        text_vi = f.read()
    tts_vietnamese(text_vi, "data/ts_fb_vi.wav")  
    jp_path = 'data/data_ja.txt'
    with open(jp_path, 'r', encoding='utf-8') as f:
        text_jp = f.read()
    tts_japanese(text_jp, "data/ts_fb_jp.wav")

if __name__ == "__main__":
    main()