import json
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker
import util_json

# Create a Faker instance
fake = Faker()

# Define the directory for the images
image_directory = 'examplesFakerGen'

# Function to clean the directory
def clean_directory(directory):
    if os.path.exists(directory):
        # Remove all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory)

# Clean the directory before generating images
clean_directory(image_directory)

# Load a default font
font = ImageFont.load_default()

def convert_png_to_jpeg(png_file_path, jpeg_file_path):
    # Open the PNG image
    with Image.open(png_file_path) as img:
        # Convert PNG to RGB (JPEG doesn't support transparency/alpha channel)
        rgb_img = img.convert('RGB')
        # Save the image in JPEG format
        rgb_img.save(jpeg_file_path, "JPEG")
    print(f"Image saved at {jpeg_file_path}")

# Function to draw a random shadow on the image
def draw_random_shadow(draw, img_size):
    shadow = Image.new('RGBA', img_size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    # Choose a random shadow type
    shadow_type = random.choice(["hand", "cup", "bottle"])

    if shadow_type == "hand":
        # Draw a hand-like shadow using random ellipses and polygons
        num_fingers = random.randint(3, 5)
        for _ in range(num_fingers):
            # Randomly generate finger shapes
            x0 = random.randint(100, 200)
            y0 = random.randint(300, 900)
            x1 = x0 + random.randint(20, 50)
            y1 = y0 + random.randint(100, 150)
            shadow_draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, random.randint(50, 150)))

    elif shadow_type == "cup":
        # Draw a cup-like shadow (circle with handle)
        x0 = random.randint(100, 500)
        y0 = random.randint(100, 600)
        r = random.randint(50, 100)
        shadow_draw.ellipse([x0, y0, x0 + r, y0 + r], fill=(0, 0, 0, random.randint(50, 150)))
        # Draw the handle
        shadow_draw.ellipse([x0 + r, y0 + int(r/2), x0 + r + int(r/2), y0 + int(1.5 * r)], outline=(0, 0, 0, random.randint(50, 150)))

    elif shadow_type == "bottle":
        # Draw a bottle-like shadow (rectangle with ellipse top)
        x0 = random.randint(100, 400)
        y0 = random.randint(100, 500)
        width = random.randint(30, 60)
        height = random.randint(100, 200)
        shadow_draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(0, 0, 0, random.randint(50, 150)))
        # Draw the top ellipse
        shadow_draw.ellipse([x0, y0 - int(width/2), x0 + width, y0 + int(width/2)], fill=(0, 0, 0, random.randint(50, 150)))

    return shadow

# Function to add random scribbles on the form
def add_random_scribbles(draw):
    num_scribbles = random.randint(3, 7)
    for _ in range(num_scribbles):
        start_x = random.randint(50, 750)
        start_y = random.randint(50, 1150)
        end_x = start_x + random.randint(-100, 100)
        end_y = start_y + random.randint(-50, 50)
        draw.line([(start_x, start_y), (end_x, end_y)], fill='black', width=random.randint(1, 3))

# Function to create and fill a medical form with random data, shadows, scribbles, and crumpling
def create_filled_medical_form(image_number):
    # Create an image with a white background
    img = Image.new('RGB', (800, 1200), color='white')
    draw = ImageDraw.Draw(img)

    # Draw form title
    draw.text((20, 20), "Medical Information Form", font=font, fill='black')

    # Draw form sections and labels
    draw.text((20, 80), "Patient Information:", font=font, fill='black')
    draw.text((20, 120), "Name:", font=font, fill='black')
    draw.text((20, 160), "Date of Birth:", font=font, fill='black')
    draw.text((20, 200), "Address:", font=font, fill='black')
    draw.text((20, 240), "Phone Number:", font=font, fill='black')

    # Generate random patient information
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
    address = fake.address().replace("\n", ", ")
    phone_number = fake.phone_number()

    # Generate fixed patient information
    #name = "John Doe"
    #dob = '1990-12-31'
    #address = '1234 My Street City 98765'
    #phone_number = '647-999-3333'

    # Fill text fields
    draw.text((160, 120), name, font=font, fill='black')
    draw.text((160, 160), dob, font=font, fill='black')
    draw.text((160, 200), address, font=font, fill='black')
    draw.text((160, 240), phone_number, font=font, fill='black')

    # Draw section for medical history
    draw.text((20, 300), "Medical History:", font=font, fill='black')
    draw.text((20, 340), "1. Do you have any chronic illnesses?", font=font, fill='black')
    draw.text((20, 380), "2. Are you currently taking any medication?", font=font, fill='black')
    draw.text((20, 420), "3. Do you have any allergies?", font=font, fill='black')

    # Draw checkboxes for "Yes" / "No" answers
    for y in [335, 375, 415]:
        draw.ellipse([(350, y), (370, y + 20)], outline='black')  # Yes checkbox
        draw.text((380, y), "Yes", font=font, fill='black')
        draw.ellipse([(450, y), (470, y + 20)], outline='black')  # No checkbox
        draw.text((480, y), "No", font=font, fill='black')

    # Randomly select "Yes" or "No" for the questions
    answers = [random.choice(["Yes", "No"]) for _ in range(3)]

    # Fill the selected answers
    for index, y in enumerate([335, 375, 415]):
        if answers[index] == "Yes":
            draw.ellipse([(350, y), (370, y + 20)], fill='black')  # Fill "Yes" circle
        else:
            draw.ellipse([(450, y), (470, y + 20)], fill='black')  # Fill "No" circle

    # Draw signature section
    draw.text((20, 500), "Signature:", font=font, fill='black')

    # Draw a random signature (simulate a handwritten signature)
    signature_start_x = 160
    signature_start_y = 500
    points = [(signature_start_x + random.randint(-5, 5),
               signature_start_y + random.randint(-5, 5)) for _ in range(20)]

    # Choose a random pen color for the signature
    signature_color = random.choice(['blue', 'red'])

    # Draw a larger signature with the chosen color
    draw.line(points, fill=signature_color, width=30)

    # Add random scribbles on the form
    add_random_scribbles(draw)

    # Add random shadow to the page
    shadow = draw_random_shadow(draw, img.size)

    # Composite the shadow onto the original image
    img = Image.alpha_composite(img.convert('RGBA'), shadow)

    # Apply a crumpled paper effect using a distortion filter
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))

    # Randomize a bit with rotation to simulate camera capture
    img = img.rotate(random.randint(-5, 5), expand=1)

    # Create a JSON Results file, which will be used later to get similarity scores
    # Values to update in the JSON template using hierarchical keys

    field_values = {
        "patient_information:name": "Hugh Herr",
        "patient_information:date_of_birth": "1985-05-15",
        "patient_information:address": "5678 Another St, City 12345",
        "patient_information:phone_number": "416-555-7777",
        "patient_information:signature": "Jane's Signature",
        "patient_information:medical_history:Do you have any chronic illnesses?": "No",
        "patient_information:medical_history:Are you currently taking any medication?": "No",
        "patient_information:medical_history:Do you have any allergies?": "Yes"
    }
    field_values["patient_information:name"] = name
    field_values["patient_information:date_of_birth"] = dob
    field_values["patient_information:address"] = address
    field_values["patient_information:phone_number"] = phone_number
    field_values["patient_information:medical_history:Do you have any chronic illnesses?"] = answers[0]
    field_values["patient_information:medical_history:Are you currently taking any medication?"] = answers[1]
    field_values["patient_information:medical_history:Do you have any allergies?"] = answers[2]

    result = util_json.fill_json_fields_with_hierarchy(field_values)

    # Save the results json file
    json_filepath = f'{image_directory}/form_{image_number}_result.json'
    with open(json_filepath, 'w') as json_file:
        json.dump(result, json_file, indent=4)
    # Save the image
    img = img.convert('RGB')  # Convert back to RGB mode before saving
    img.save(f'{image_directory}/form_{image_number}.png')

    # Save a jpeg copy
    convert_png_to_jpeg(f'{image_directory}/form_{image_number}.png', f'{image_directory}/form_{image_number}.jpeg')


# Generate 100 filled medical forms with realistic shadows, scribbles, and crumpling
for i in range(1, 101):
    create_filled_medical_form(i)

print("Filled medical forms with realistic shadows and scribbles generated successfully!")
