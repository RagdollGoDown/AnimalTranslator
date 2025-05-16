from scipy.io import wavfile
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

import os

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

def query_qwen2(samplerate, data, question="What does this sound like?"):
    # Load and process the audio
    inputs = processor(data, question, return_tensors="pt", sampling_rate=samplerate)

    # Generate caption
    outputs = model.generate(**inputs)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return caption

def main():
    audio_file_path = os.getcwd() + "\\assets\\audio\\hey_man.wav"

    samplerate, data = wavfile.read(audio_file_path)

    question = "What do you hear?"

    caption = query_qwen2(samplerate, data, question)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()