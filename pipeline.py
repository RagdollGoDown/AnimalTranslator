from test_blip import query_blip
from test_whisper import query_whisper 
from mistral import ask_mistral, image_and_text_to_text

import os
from PIL import Image
import librosa

def query_image(image_path, questions):
    # Load and process the image
    answers = image_and_text_to_text(image_path, questions)
    return answers

def query_image_blip(image, questions):
    # Load and process the image
    answers = []

    for question in questions:
        answers.append(query_blip(image, question))
    return answers


def speech_recognition(audio):
    result = query_whisper(audio)
    return result

def pipeline_full(image_path, audio) -> str:

    image_questions = [
        "Give me a descritpition of the animal in the image, it's emotional state and what it's doing.",
    ]

    image = Image.open(image_path)

    # image -> classify
    answers_image = query_image(image_path, questions=image_questions)
    #answers_image = query_blip(image,image_questions)
    
    image_info_block = ""
    for question, answer in zip(image_questions, answers_image):
        image_info_block += f"Question: {question}\nAnswer: {answer}\n\n"
    print("Image information block:", image_info_block)
    

    # audio -> transcribe
    result = speech_recognition(audio)
    print("Animal speech detected:", result)

    # information aggregation
    final_text_question = f"""
    You are an animal translator.

    The information retrieved from the image is:\n
    {image_info_block}
    The information retrieved from the sound is:\n
    {result}

    Imagine yourself as a translator for animals.
    You have information from the image and the sound. Of course, you can't understand the animal,
    but you are going to act like a confident translator and give a clear and plausible answer of what the animal is saying.

    Keep in mind the information from the sound is just speech recognition, not the actual animal speech.
    So it doesn't actually mean anything and you MUST NOT mention it. 

    Give a short and simple answer, like a translator would do and cooborate it with the information from the image:
    \n
    """
    
    final_text = ask_mistral(final_text_question)

    return final_text

def pipeline_audio(audio) -> str:
    # audio -> transcribe
    result = speech_recognition(audio)
    print("Animal speech detected:", result)

    # information aggregation
    final_text_question = f"""
    The information retrieved from the sound is:\n
    {result}

    You are analyzing an animal in a sound. 
    Do not mention it and only use it as inspiration for your answer.
    You do not have to actually tell the truth about the animal, just give a plausible answer with confidence.
    \n
    """
    
    final_text = ask_mistral(final_text_question)

    return final_text

def pipeline_image(image_path) -> str:
    image_questions = [
        "Give me a descritpition of the animal in the image, it's emotional state and what it's doing.",
    ]

    # image -> classify
    answers_image = query_image(image_path, questions=image_questions)
    #answers_image = query_blip(image,image_questions)
    
    image_info_block = ""
    for question, answer in zip(image_questions, answers_image):
        image_info_block += f"Question: {question}\nAnswer: {answer}\n\n"
    print("Image information block:", image_info_block)

    # information aggregation
    final_text_question = f"""
    The information retrieved from the image is:\n
    {image_info_block}
    
    You are analyzing an animal in an image. 
    Give a plausible answer with confidence.
    Keep it short and simple.
    \n
    """
    
    final_text = ask_mistral(final_text_question)

    return final_text


def main():
    # Example usage
    image_path = os.getcwd() + "\\assets\\images\\monkey.png"
    audio_path = os.getcwd() + "\\assets\\audio\\monkey_noises.wav"

    # Load the image
    #image = Image.open(image_path)

    # Load the audio
    audio, sr = librosa.load(audio_path, sr=16000)

    final_text = pipeline_full(image_path, audio)
    print("Final text:", final_text)

if __name__ == "__main__":
    main()