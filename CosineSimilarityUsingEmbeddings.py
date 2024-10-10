import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Step 1: Convert JSON objects to strings (or a flattened form)
def json_to_string(json_obj):
    return json.dumps(json_obj, sort_keys=True)

def calculatecosingsimilarity(json1, json2):
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
