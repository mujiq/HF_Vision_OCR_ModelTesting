import json, os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Step 1: Convert JSON objects to strings (or a flattened form)
def json_to_string(json_obj):
    return json.dumps(json_obj, sort_keys=True)


def read_json_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return f"Error: The file at {file_path} does not exist."

    try:
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Convert the loaded JSON data back to a JSON string
        json_string = json.dumps(json_data, indent=4)
        return json_string

    except json.JSONDecodeError as e:
        return f"Error: Failed to decode JSON. {str(e)}"

    except Exception as e:
        return f"Error: {str(e)}"

def calculatecosingsimilarity(json_obj1, json_obj2):
    text1 = json_to_string(json_obj1)
    text2 = json_to_string(json_obj2)

    # Step 2: Get embeddings for the texts using a pre-trained model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose another model if needed
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]

    # Step 3: Calculate cosine similarity
    similarity_score = cosine_similarity([embedding1], [embedding2])

    # Output the similarity score
    similarity_score_value = similarity_score[0][0]
    print(f"Similarity Score: {similarity_score_value}")
    return similarity_score_value

# Function to fill values in the JSON template using hierarchical keys
def fill_json_fields_with_hierarchy( field_values):
    # Example template JSON structure
    json_template = {
        "form": "Medical Information Form",
        "patient_information": {
            "name": "John Doe",
            "date_of_birth": "1990-12-31",
            "address": "1234 My Street City 98765",
            "phone_number": "647-999-3333",
            "medical_history": [
                {
                    "question": "Do you have any chronic illnesses?",
                    "answer": "N/A"
                },
                {
                    "question": "Are you currently taking any medication?",
                    "answer": "N/A"
                },
                {
                    "question": "Do you have any allergies?",
                    "answer": "N/A"
                }
            ],
            "signature": "Signature"
        }
    }

    def fill_fields(data, key_path, value):
        keys = key_path.split(":")
        current_key = keys[0]

        # If the current key points to a list (medical_history), match the question and update the answer
        if isinstance(data, dict) and current_key == "medical_history":
            question_to_match = keys[1]
            child = data.get("medical_history", [])

            for item in child:
                if item.get("question") == question_to_match:
                    item["answer"] = value  # Update the answer field in matching entry

        # If the current key is found in the dictionary, traverse further if needed
        elif isinstance(data, dict) and current_key in data:
            if len(keys) == 1:
                data[current_key] = value
            else:
                fill_fields(data[current_key], ":".join(keys[1:]), value)



    # Iterate over the field values and update the JSON template
    for key_path, value in field_values.items():
        fill_fields(json_template, key_path, value)

    return json_template



if __name__ == '__main__':
    json1 = {
        "form": "Medical Information Form",
        "patient_information": {
            "name": "John Doe",
            "date_of_birth": "1990-12-31",
            "address": "1234 My Street City 98765",
            "phone_number": "647-999-3333",
            "medical_history": [
                {
                    "question": "Do you have any chronic illnesses?",
                    "answer": "Yes"
                },
                {
                    "question": "Are you currently taking any medication?",
                    "answer": "Yes"
                },
                {
                    "question": "Do you have any allergies?",
                    "answer": "No"
                }
            ],
            "signature": "Signature"
        }
    }
    json2 = {
        "form": "Medical Information Form",
        "patient_information": {
            "name": "John Doe 2",
            "date_of_birth": "1990-12-31",
            "address": "1234 My Street City 987654",
            "phone_number": "647-999-3333",
            "medical_history": [
                {
                    "question": "Do you have any chronic illnesses?",
                    "answer": "Yes"
                },
                {
                    "question": "Are you currently taking any medication?",
                    "answer": "Yes"
                },
                {
                    "question": "Do you have any allergies?",
                    "answer": "No"
                }
            ],
            "signature": "Signature"
        }
    }
    similarity = calculatecosingsimilarity(json1, json2)
    #similarity = calculate_similarity(json1, json2)
    #print(f"Field Similarity: {similarity['field_similarity']:.2f}%")
    #print(f"Content Similarity: {similarity['content_similarity']:.2f}%")
    #print(f"Overall Similarity: {similarity['overall_similarity']:.2f}%")


    # Values to update in the JSON template using hierarchical keys
    name = "John Doe"
    dob = "1979-01-30"

    field_values = {
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


    result = fill_json_fields_with_hierarchy(field_values)

