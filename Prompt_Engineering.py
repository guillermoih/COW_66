# +
#import os
#os.system("pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html")
# -

# # Imports

# +
import json, requests
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers import StableDiffusionXLPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
# -

# # Models load

# ## Stable Diffusion XL

print("#"*5, "Loading Stable Diffusion XL", "#"*5)

# +
STABLE_DIFF_PIPELINE = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

STABLE_DIFF_PIPELINE.to("cuda")
# -

# ## BLIP

print("#"*5, "Loading BLIP", "#"*5)

BLIP_PROCCESOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# # Caption generation using Blip

def generate_caption(img):
    inputs = BLIP_PROCCESOR(img, return_tensors="pt")

    out = BLIP_MODEL.generate(**inputs)
    return BLIP_PROCCESOR.decode(out[0], skip_special_tokens=True)


# # Sentence similarity with DistilBERT

# +
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
api_token = "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"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# -

def sentence_similarity(input_sentence, sentences):
    data = query(
        {
            "inputs": {
                "source_sentence": input_sentence,
                "sentences": sentences
            }
        })
    
    return data


# # Generating original batch of images

prompt = "Hello Kitty in the space riding a bicycle"
images = STABLE_DIFF_PIPELINE(prompt=prompt, num_images_per_prompt=5,
                             output_type='pil').images

# # Calculate caption and distance similarities

# +
captions = []
for i, img in enumerate(images):
    img.save("test_" + str(i) + ".jpg")
    
    print("Processing image", i)
    caption = generate_caption(img)
    print("Generated caption:", caption)
    captions.append(caption)

similarities = sentence_similarity(prompt, captions)
# -

for caption, similarity in zip(captions, similarities):
    print("Similarity for:", caption)
    print(similarity)
    print("\n", "-"*10)
