import spaces
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import numpy as np
import os
from datetime import datetime
import subprocess

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Florence model
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to(device).eval()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

# Initialize Qwen2-VL-2B model
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype="auto").to(device).eval()
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

@spaces.GPU
def florence_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    return parsed_answer["<MORE_DETAILED_CAPTION>"]

def array_to_image_path(image_array):
    img = Image.fromarray(np.uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    full_path = os.path.abspath(filename)
    return full_path

@spaces.GPU
def qwen_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
    
    image_path = array_to_image_path(np.array(image))
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Describe this image in great detail in one paragraph."},
            ],
        }
    ]
    
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]