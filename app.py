from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForConditionalGeneration
model = BlipForConditionalGeneration.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

import requests
image_url = "https://hips.hearstapps.com/hmg-prod/images/edc-boardgames-2-1585943669.jpg?crop=0.503xw:1.00xh;0,0&resize=1200:*"
response = requests.get(image_url)


IMAGE_PATH = "downloaded_image.jpg"
if response.status_code == 200:
    with open(IMAGE_PATH, "wb") as file:
        file.write(response.content)


text = "the main object of the photograph"
inputs = processor(image, text, return_tensors="pt")

println(inputs)
out = model.generate(**inputs)
println(out)

print(processor.decode(out[0], skip_special_tokens=True))
inputs = processor(image,return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
