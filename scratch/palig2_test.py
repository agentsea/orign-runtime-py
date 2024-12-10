from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

model_id = "google/paligemma2-3b-pt-896"

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image1 = load_image(url1)

url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Motorboat_at_Kankaria_lake.JPG/1280px-Motorboat_at_Kankaria_lake.JPG"
image2 = load_image(url2)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Leaving the prompt blank for pre-trained models
prompt = "Describe these images in detail"
model_inputs = processor(text=prompt, images=[[image1, image2]], return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

# print("model_inputs: ", model_inputs)

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print("result: ", decoded)
