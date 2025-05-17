"""
Using https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503

"""

import requests
from keys import HUGGING_FACE_KEY
from keys import MISTRAL_API_KEY
import time
import base64
import requests

import os
from mistralai import Mistral

api_key = MISTRAL_API_KEY
model1 = "mistral-large-latest"
model2 = "pixtral-12b-2409" 

client = Mistral(api_key=api_key)


def ask_mistral(question):
    chat_response = client.chat.complete(
        model = model1,
        messages = [
            {
                "role": "user",
                "content": f"{question}",
            },
        ]
    )
    return chat_response.choices[0].message.content

def image_and_text_to_text(image_path, questions):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"
    # Define the messages for the chat
    response = []
    for question in questions:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": image_url
                    }
                ]
            }
        ]

        # Get the chat response
        chat_response = client.chat.complete(
            model=model2,
            messages=messages
        )
        response.append(chat_response.choices[0].message.content)
    return response