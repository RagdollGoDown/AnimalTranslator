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
    The information retrieved from the image is:\n
    {image_info_block}
    The information retrieved from the sound is:\n
    {result}

    You are analyzing an animal in an image and a sound. We are going to give information retrieved from both.
    The informat from the image is a reliable and can appear in your answer, but the information from the sound is not reliable.
    The speech recognition should only serve as inspiration for your answer and not appear in it.
    Give a plausible answer with confidence.
    Keep it short and simple.
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
    The information retrieved from the sound is a speech recognition result. 
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

    # Load and process the image
    answers_image = query_image(image_path, questions=image_questions)
    
    image_info_block = ""
    for question, answer in zip(image_questions, answers_image):
        image_info_block += f"Question: {question}\nAnswer: {answer}\n\n"
    print("Image information block:", image_info_block)

    # information aggregation
    final_text_question = f"""
    The information retrieved from the image is:\n
    {image_info_block}
    
    You are analyzing an animal in an image. 
    Do not mention it and only use it as inspiration for your answer.
    You do not have to actually tell the truth about the animal, just give a plausible answer with confidence.
    \n
    """
    
    final_text = ask_mistral(final_text_question)

    return final_text


def main():
    # Example usage
    image_path = os.getcwd() + "\\assets\\images\\monkey.png"
    audio_path = os.getcwd() + "\\assets\\audio\\ylan_barking.wav"

    # Load the image
    #image = Image.open(image_path)

    # Load the audio
    audio, sr = librosa.load(audio_path, sr=16000)

    final_text = pipeline_full(image_path, audio)
    print("Final text:", final_text)

if __name__ == "__main__":
    main()