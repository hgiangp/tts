from transformers import AutoProcessor, SeamlessM4Tv2Model

model_id = "facebook/seamless-m4t-v2-large"
local_dir = "./models/seamless-m4t-v2-large"

# Important: trust_remote_code is required
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = SeamlessM4Tv2Model.from_pretrained(model_id, trust_remote_code=True)

# Save locally
processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("Model and processor downloaded successfully.")
