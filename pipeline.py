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
    print("Animal found:", caption)

    # audio -> transcribe
    result = speech_recognition(audio)
    print("Animal speech detected:", result)

    # information aggregation
    final_text_question = f"""
    We have these informations :\n
    Animal found: {caption} \n Animal speach detected: {result} \n
    You are translating what the animal is saying. This information might not suffice 
    so you will need to use your imagination. It does not need to be true.

    Also assume the classification is true, but the translation is just human speechrecognition applied to an animal.
    so it should just serve as a hint to what the animal is saying. Do not mention it in your answer.
    Be confident in you're answer and give a single answer. Keep it short and simple.
    \n

    """
    
    final_text = mistral.ask_mistral(final_text_question)

    return final_text['choices'][0]['message']['content']


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