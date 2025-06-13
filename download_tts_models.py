import os
import shutil
from TTS.api import TTS
from huggingface_hub import snapshot_download
from params import * 

def download_coqui_model(model_name: str, target_dir: str):
    print(f"Downloading Coqui TTS model: {model_name}")
    tts = TTS(model_name=model_name)
    cache_dir = os.path.expanduser("~/.local/share/tts/")
    model_dir = os.path.join(cache_dir, *model_name.split("/"))

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found: {model_dir}")

    if os.path.exists(target_dir):
        print(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    shutil.copytree(model_dir, target_dir)
    print(f"Coqui model copied to: {target_dir}")

def download_mms_model(repo_id: str, target_dir: str):
    print(f"Downloading Hugging Face MMS model: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"MMS model downloaded to: {target_dir}")

if __name__ == "__main__":
    # Coqui Japanese TTS model
    coqui_model = TTS_JA_MODEL_NAME
    coqui_target = TTS_JA_MODEL_DIR

    # Hugging Face MMS Vietnamese TTS model
    mms_repo = TTS_VI_MODEL_NAME
    mms_target = TTS_VI_MODEL_DIR

    # Download both
    download_coqui_model(coqui_model, coqui_target)
    download_mms_model(mms_repo, mms_target)
