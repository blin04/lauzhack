# import os
# from openai import OpenAI
#  
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
# )
#  
# completion = client.chat.completions.create(
#     model="moonshotai/Kimi-K2-Instruct-0905",
#     messages=[
#         {
#             "role": "user",
#             "content": "Generate a poem about the sea."
#         }
#     ],
# )
#  
# print(completion.choices[0].message)

# Load model directly
from transformers import Sam3Processor, Sam3Model
from PIL import Image
import torch
import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


device = "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")


image = Image.open("image.png")

inputs = processor(images=image, text="car", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")

masked_image = overlay_masks(image, results["masks"])
masked_image.save("segmented_image.png")
