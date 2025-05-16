import torch
from transformers import AutoModel, AutoProcessor, pipeline
import librosa
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModel.from_pretrained("sail-rvc/Donald_Trump__RVC_v2_")

processor = AutoProcessor.from_pretrained("sail-rvc/Donald_Trump__RVC_v2_")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load a sample audio file
sample_path = os.getcwd() + "\\assets\\audio\\ylan_barking.wav"
sample, sr = librosa.load(sample_path, sr=16000)

result = pipe(sample)
print(result["text"])