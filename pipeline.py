from test_blip import query_blip
from test_whisper import query_whisper 
import mistral

import os
from PIL import Image
import librosa

def query_image(image, question="How does this dog feel?"):
    # Load and process the image
    caption = query_blip(image, question)
    return caption

def speech_recognition(audio):
    result = query_whisper(audio)
    return result

def pipeline(image, audio) -> str:

    # image -> classify
    caption = query_image(image, "What is the animal in the image?")

    # audio -> transcribe
    result = speech_recognition(audio)

    # information aggregation
    final_text_question = f"Animal found: {caption} \n Animal feeling: {result}"
    
    final_text = mistral.ask_mistral(final_text_question)

    return final_text


def main():
    # Example usage
    image_path = os.getcwd() + "\\assets\\images\\monkey.png"
    audio_path = os.getcwd() + "\\assets\\audio\\ylan_barking.wav"

    # Load the image
    image = Image.open(image_path)

    # Load the audio
    audio, sr = librosa.load(audio_path, sr=16000)

    final_text = pipeline(image, audio)
    print("Final text:", final_text)

if __name__ == "__main__":
    main()