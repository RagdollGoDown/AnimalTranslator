# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

def query_blip(image_path):
    # Load and process the image
    image = processor(images=image_path, return_tensors="pt").pixel_values

    # Generate caption
    outputs = model.generate(image)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return caption

def main():
    # Example usage
    image_path = "assets/images/happy_dog.jpeg"  # Replace with your image path
    caption = query_blip(image_path)
    print("Generated Caption:", caption)