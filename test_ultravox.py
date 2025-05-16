import librosa
import os
from transformers import AutoModel
model = AutoModel.from_pretrained("fixie-ai/ultravox-v0_4_1-llama-3_1-8b", trust_remote_code=True)

def query_ultravox(audio, question="What does this sound like?"):
    # Load and process the audio

    # Generate caption
    outputs = model.generate(**audio)
    
    return outputs

def main():
    path = os.getcwd() + "\\assets\\audio\\hey_man.wav"  # TODO: pass the audio here
    audio, sr = librosa.load(path, sr=16000)

    question = "What do you hear?"
    caption = query_ultravox(audio, question)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()