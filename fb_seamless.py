from transformers import AutoProcessor, SeamlessM4Tv2Model
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torchaudio, torch
import scipy
import warnings

warnings.filterwarnings(category=FutureWarning, action="ignore")

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
model_dir = "models/seamless-m4t-v2-large"

def text_to_speech(text, src_lang="vie", tgt_lang="jpn", output_path="tts.wav"): 
    processor = AutoProcessor.from_pretrained(model_dir)
    model = SeamlessM4Tv2Model.from_pretrained(model_dir).to(device)
    # Prepare input and move tensors to device
    text_inputs = processor(
        text=text,
        src_lang=src_lang,
        return_tensors="pt"
    )
    # text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # Generate audio and move output to CPU for saving
    audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang)[0].to("cpu").numpy().squeeze()

    # Save to WAV file
    sample_rate = model.config.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array_from_text)

def speech_to_speech(wav_path, src_lang="vie", tgt_lang="jpn", output_path="tts.wav"): 
    processor = AutoProcessor.from_pretrained(model_dir)
    model = SeamlessM4Tv2Model.from_pretrained(model_dir).to(device)
    # from audio
    audio, orig_freq =  torchaudio.load(wav_path)
    audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio, src_lang=src_lang, return_tensors="pt")
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()

    sample_rate = model.config.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array_from_audio)

def speech_to_text(wav_path="tts_test/meian_0005.wav", src_lang="jpn", tgt_lang="jpn", output_path="tts_test/stt_jpn_jpn.txt"): 
    processor = AutoProcessor.from_pretrained(model_dir)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_dir)
    
    speech, sr = torchaudio.load(wav_path)

    if sr != 16000:
        speech = torchaudio.functional.resample(speech, sr, 16000)

    inputs = processor(audios=speech.squeeze(), return_tensors="pt", src_lang=src_lang)
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, tgt_lang=tgt_lang)

    transcript = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    # transcript = processor.decode(generated_tokens[0], skip_special_tokens=True)
    print("Transcribed:", transcript)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

def test_text_to_speech(): 
    vi_path = 'tts_test/data_vi.txt'
    with open(vi_path, 'r', encoding='utf-8') as f:
        text_vi = f.read()
        
    jp_path = 'tts_test/data_ja.txt'
    with open(jp_path, 'r', encoding='utf-8') as f:
        text_jp = f.read()
        
    text_jp_1 = "申し訳ございませんでした。今後は十分注意するようにします。 問題があった場合は、内容と対策をあわせて記載し、ご報告するようにします。"
    text_vi_1 = "Saito: Vậy à, tôi hiểu rồi. Từ nay về sau, hãy chắc chắn rằng các vấn đề như chậm trễ cần được ghi rõ trong báo cáo tuần. Nếu cứ triển khai mà không ai nhận ra thì sẽ trở thành vấn đề nghiêm trọng sau này, cả hai bên sẽ gặp khó khăn, nên mong các bạn lưu ý."
    # text_to_speech(text_vi, "vie", "jpn", "./tts_test/tts_vie_jpn.wav")
    # text_to_speech(text_vi, "vie", "vie", "./tts_test/tts_vie_vie.wav")
    # text_to_speech(text_jp, "jpn", "jpn", "./tts_test/tts_jpn_jpn.wav")
    # text_to_speech(text_jp, "jpn", "vie", "./tts_test/tts_jpn_vie.wav")
    # text_to_speech(text_jp_1, "jpn", "jpn", "./tts_test/tts_jpn_jpn_1.wav")
    # text_to_speech(text_jp_1, "jpn", "vie", "./tts_test/tts_jpn_vie_1.wav")
    text_to_speech(text_vi_1, "vie", "jpn", "./tts_test/tts_vie_jpn_1.wav")
    text_to_speech(text_vi_1, "vie", "vie", "./tts_test/tts_vie_vie_1.wav")
    
def test_speech_to_speech(): 
    # speech_to_speech("./tts_test/meian_0005.wav", "jpn", "vie", "./tts_test/sts_jpn_vie.wav")
    speech_to_speech("./tts_test/meian_0005.wav", "jpn", "jpn", "./tts_test/sts_jpn_jpn.wav")

def test_speech_to_text(): 
    speech_to_text()

    
if __name__=="__main__": 
    # speech_to_text()
    # speech_to_speech()
    # test_text_to_speech()
    test_speech_to_speech()
    # test_speech_to_text()