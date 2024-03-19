import os
os.system("pip install -U sentence-transformers")

# +
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusers import StableDiffusionXLPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# +
pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

pipeline_text2image.to("cuda")
# -

prompt = "Hello Kitty in the space riding a bike"
images = pipeline_text2image(prompt=prompt, num_images_per_prompt=1,
                             output_type='pil').images

for i, img in enumerate(images):
    img.save("test_" + str(i) + ".jpg")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# +
inputs = processor(images[0], return_tensors="pt")

out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)

# +
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_1= model.encode(prompt, convert_to_tensor=True)
embedding_2 = model.encode(caption, convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embedding_1, embedding_2)

print(similarity)
