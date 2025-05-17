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
model = "mistral-large-latest"

client = Mistral(api_key=api_key)


def ask_mistral(question):
    chat_response = client.chat.complete(
        model = model,
        messages = [
            {
                "role": "user",
                "content": f"{question}",
            },
        ]
    )
    return chat_response.choices[0].message.content

def image_and_text_to_text(image_path, questions):
    # Define the messages for the chat
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": questions
                },
                {
                    "type": "image_url",
                    "image_url": image_path
                }
            ]
        }
    ]

    # Get the chat response
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    # Print the content of the response
    return chat_response.choices[0].message.content