from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel
import torch

import os
print(os.environ.get('CUDA_PATH'))

model_id = "microsoft/Phi-3.5-vision-instruct"

'''model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()

'''
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="cuda",
  trust_remote_code=True,
  torch_dtype="auto",
  #use_flash_attn=False
)

processor = AutoProcessor.from_pretrained(model_id,
  trust_remote_code=True,
  num_crops=16  # Use 16 for single-frame tasks with complex images
)

json_template = {
    "form": "Medical Information Form",
    "patient_information": {
        "name": "John Doe",
        "date_of_birth": "1990-12-31",
        "address": "1234 My Street City 98765",
        "phone_number": "647-999-3333",
        "medical_history": [
            {
                "question": "Question1",
                "answer": "value"
            },
            {
                "question": "Question2",
                "answer": "value"
            },
        ],
        "signature": "Signature"
    }
}

images = []
placeholder = ""

# Load local image
local_image_path = "./examplesFakerGen/form_5.jpeg"  # Replace with your actual image path
images.append(Image.open(local_image_path))
placeholder += "<|image_1|>\n"

messages = [
    {"role": "user", "content": placeholder+f"Extract information from this form image, provide output in the following strict json format. response={json_template}"},
]


def generate_response(messages, images):
    prompt = processor.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 10000,  # Increased for longer, more detailed responses
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50,
    }

    generate_ids = model.generate(**inputs,
      eos_token_id=processor.tokenizer.eos_token_id,
      **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False)[0]

    return response

response = generate_response(messages, images)


