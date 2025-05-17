# Load model directly
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
import os

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def query_blip(image, question="How does this dog feel?"):
    # Load and process the image
    inputs = processor(images=image,text=question ,return_tensors="pt")

    # Generate caption
    outputs = model.generate(**inputs)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return caption

def main():
    # Example usage
    image_path = os.getcwd() + "\\assets\\images\\monkey.png"
    # Load the image
    image = Image.open(image_path)

    question = "What is the animal in the image?"

    caption = query_blip(image, question)
    print("Animal found:", caption)

    question = "How is the animal feeling?"

    caption = query_blip(image, question)
    print("Animal feeling:", caption)

    question = "What color is this animal?"

    caption = query_blip(image, question)
    print("Animal color:", caption)

if __name__ == "__main__":
    main()