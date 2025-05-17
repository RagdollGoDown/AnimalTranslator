import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def query_whisper(audio):
    
    result = pipe(audio)
    return result["text"]
    

def main():
    # Load a sample audio file
    sample_path = os.getcwd() + "\\assets\\audio\\ylan_barking.wav"
    sample, sr = librosa.load(sample_path, sr=16000)
    
    result = query_whisper(sample)
    print(result)

if __name__ == "__main__":
    main()