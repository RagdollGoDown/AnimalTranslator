"""
Using https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503

"""

import requests
from keys import HUGGING_FACE_KEY
import time
import base64





import requests

API_URL =  "https://router.huggingface.co/nebius/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {HUGGING_FACE_KEY}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

start_time = time.time()

### --- Image-Text to Text
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
# # -- or using local image
# image_path = "tmp_photo.jpg"
# with open(image_path, "rb") as f:
#     base64_image = base64.b64encode(f.read()).decode("utf-8")
# image_url = f"data:image/jpeg;base64,{base64_image}"
'''
response = query({
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            ]
        }
    ],
    "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
})
'''

'''
if n_words > 0:
    question += f" (in {n_words} words or less)"
    response = query({
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ],
    "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    })
    '''


def ask_mistral(question):
    response = query({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{question}"
                    }
                ]
            }
        ],
        "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        })
    return response


def image_and_text_to_text(image_path, question):
    with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"
            response = query({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{question}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        ]
                    }
                ],
                "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
            })
    return response






#input_text = "You are translating what the dog is saying : We have these informations :"
#response_main = ask_mistral(input_text)

input_text = "What is in this image ?"
image_path = "assets/images/happy_dog.jpeg"
response_main = image_and_text_to_text(image_path, input_text)


print(response_main["choices"][0]["message"]["content"])
print("Time taken :", time.time() - start_time)