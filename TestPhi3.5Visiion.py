from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel
import torch

import os
print(os.environ.get('CUDA_PATH'))

model_id = "microsoft/Phi-3.5-vision-instruct"
model_name = "Phi-3.5-vision-instruct"

#model_id = "microsoft/Phi-3-vision-128k-instruct"
#model_name = "Phi-3.5-vision-128k-instruct"

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
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
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


import os, json, csv, time, util_json
from datetime import datetime

input_dir = "examplesFakerGen"

# Generate a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"results/{model_name}-results_{timestamp}.csv"

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    fieldnames = ['Model','Image_filename', 'Extracted-text', 'Inference-time','Similarity-Score']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    for i in range(1, 101):
        file_name = f"form_{i}.jpeg"
        input_path = os.path.join(input_dir, file_name)
        result_path = os.path.join(input_dir, f"form_{i}_result.json")
        expected_response = util_json.read_json_file(result_path)

        # check if the file exists
        if os.path.exists(input_path):
            # open the image
            pixel_values = load_image(input_path, max_num=12).to(torch.bfloat16).cuda()

            try:
                images = []
                placeholder = ""

                with Image.open(input_path) as img:

                    # Measure inference time
                    start_time = time.time()

                    #images.append(pixel_values) # Phi handles image manipulation itself no pre-processing needed
                    images.append(img)
                    placeholder += "<|image_1|>\n"

                    messages = [
                        {"role": "user",
                         "content": placeholder + f"Extract information from this form image, provide output in the following strict json format. donot add any extra response or json tag {json_template}"},
                    ]

                    actual_response = generate_response(messages, images)
                    inference_time = time.time() - start_time


                    score = util_json.calculatecosingsimilarity(expected_response, actual_response)
                    print(f'User: {messages[0]}\nAssistant: {actual_response}\nScore: {score}\n')


                    # Convert the JSON data to a string
                    json_str = json.dumps(actual_response)

                    # Write the filename and JSON data to the CSV
                    writer.writerow({'Model': model_name, 'Image_filename': file_name, 'Extracted-text': json_str,
                                     'Inference-time': inference_time, 'Similarity-Score': score})

                    print(f"Processed {file_name}.")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

exit()


# Load local image
local_image_path = "./examplesFakerGen/form_5.jpeg"  # Replace with your actual image path
images.append(Image.open(local_image_path))
placeholder += "<|image_1|>\n"

messages = [
    {"role": "user", "content": placeholder+f"Extract information from this form image, provide output in the following strict json format. response={json_template}"},
]

response = generate_response(messages, images)


