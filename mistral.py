#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.


Usage:

```python
python echobot.py
```

Press Ctrl-C on the command line to stop the bot.

"""

import logging
import os
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from keys import TELEGRAM_KEY
from keys import HUGGING_FACE_KEY
import requests
import base64

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

os.makedirs("downloads", exist_ok=True)

API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {HUGGING_FACE_KEY}",
}

last_image_id = ""

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    msg = update.message

    if msg.photo:
        photo: telegram.PhotoSize = msg.photo[-1]
        if msg.caption:
            await msg.reply_text(f"Vous avez envoyé une photo avec la légende : « {msg.caption} »")

            file = await photo.get_file()  

            filename = f"assets/images/{photo.file_unique_id}.jpg"

            await file.download_to_drive(custom_path=filename)
            image_path = filename
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
                                "text": msg.caption
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
            
            await msg.reply_text(response["choices"][0]["message"]["content"])
        else:
            await msg.reply_text("Vous avez envoyé une photo sans légende.")
        
        last_image_id = photo.file_unique_id
        


    elif msg.text:
        await msg.reply_text(f"Vous avez envoyé un texte : « {msg.text} »")

    elif msg.audio or msg.voice:
        audio = None
        if msg.audio:
            audio = msg.audio
        else:
            audio = msg.voice
        
        audio_file = await msg.get_file()
        tmp_file = f"assets/audio/{audio.file_unique_id}.wav"
        await audio_file.download_to_drive(tmp_file)
        await msg.reply_text(f"Vous avez envoyé un audio ")

    else:
        await msg.reply_text("Type de message non géré par echo.")


    #await update.message.reply_text(update.message.text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_KEY).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()