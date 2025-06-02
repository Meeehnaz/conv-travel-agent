import json, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

openai.api_key= os.getenv("OPEN_AI_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

def load_embeddings(filename):
    with open(filename, "r") as f:
        return json.load(f)

category_embeddings = load_embeddings("category_embeddings.json")
condition_embeddings = load_embeddings("condition_embeddings.json")

# Convert embedding lists to numpy arrays
category_embeddings = {k: np.array(v) for k, v in category_embeddings.items()}
condition_embeddings = {k: np.array(v) for k, v in condition_embeddings.items()}

# Get embedding from OpenAI for input text
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response['data'][0]['embedding'])

# Find matched key from embeddings
def best_match(user_vector, embedding_dict, threshold=0.5):
    best_key = None
    best_score = -1
    for key, emb in embedding_dict.items():
        score = cosine_similarity([user_vector], [emb])[0][0]
        # print(f"Checking {key}: score={score}")
        if score > best_score:
            best_score = score
            best_key = key
    # print(f"Best match: {best_key} with score {best_score}")
    if best_score >= threshold:
        return best_key
    return None


def map_user_to_geoapify(user_data):
    categories = set()
    conditions = set()

    # Field-specific fallback categories
    field_fallbacks = {
        "cuisine": "catering.restaurant",
        "food": "catering.restaurant",
        "stay_type": "accommodation.hotel",
        "activities": "tourism.attraction",
        "preferred_transport_mode": "rental.car",
        "special_requests": "access.yes",
    }

    matched_fields = set()

    special_requests = user_data.get("special_requests")
    if special_requests:
        special_emb = get_embedding(special_requests)
        cond_match = best_match(special_emb, condition_embeddings)
        if cond_match:
            conditions.add(cond_match)
        else:
            cat_match = best_match(special_emb, category_embeddings)
            if cat_match:
                categories.add(cat_match)

    skip_keys = {
        "email", "name", "travelers",
        "return_date", "departure_date", "origin", "destination"
    }

    for key, val in user_data.items():
        if key in skip_keys:
            continue

        if isinstance(val, (int, float)):
            val = str(val)
        if len(str(val).split()) < 3:
            val = f"{key.replace('_', ' ')} is {val}"

        emb = get_embedding(str(val))
        cat_match = best_match(emb, category_embeddings)

        if cat_match:
            categories.add(cat_match)
            matched_fields.add(key)
        else:
            fallback = field_fallbacks.get(key)
            if fallback:
                print(f"No match for '{key}'. Using fallback: {fallback}")
                categories.add(fallback)

    return list(categories), list(conditions)


def geocode_location(location):
    url = (
        f"https://api.geoapify.com/v1/geocode/search?"
        f"text={location}&apiKey={GEOAPIFY_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['features']:
            coords = data['features'][0]['geometry']['coordinates'] 
            return coords[1], coords[0] 
    print(f"Failed to geocode location: {location}")
    return None, None


def query_geoapify_places(categories, lat, lon, per_category_limit=2, radius=10000):
    all_results = []

    for cat in categories:
        url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories={cat}"
            f"&filter=circle:{lon},{lat},{radius}"
            f"&limit={per_category_limit}"

            f"&apiKey={GEOAPIFY_API_KEY}"
        )
        print(f"\nQuerying category: {cat}")
        response = requests.get(url)
        if response.status_code == 200:
            places = response.json().get("features", [])
            for i, place in enumerate(places):
                props = place["properties"]
                name = props.get('name', 'Unnamed')
                address = props.get('formatted', '')
                print(f" - {name} ({cat}) - {address}")
                all_results.append({
                    "name": name,
                    "category": cat,
                    "address": address
                })
        else:
            print(f"Request failed for category: {cat} - {response.text}")

    return all_results



def generate_email_draft(name, destination, dep_date, ret_date, places):
    place_lines = "\n".join(
        [f"- {p['name']} ({p['category']}): {p['address']}" for p in places]
    )

    prompt = f"""
Create a personalized travel itinerary for {name}, who is traveling to {destination} from {dep_date} to {ret_date}.
Here are the suggested places to include:

{place_lines}

The plan should be 2-3 paragraphs, friendly, and helpful. It should mention the best way to organize their trip.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content'].strip()

def send_email_to_n8n(email, subject, body):
    try:
        n8n_webhook_url = "http://localhost:5678/webhook-test/travel-plan"
        
        # Try different payload formats
        payload =  {"to": email, "subject": subject, "text": body}
        
        print(f"\nSending Email")
        
        # print(json.dumps(payload, indent=2))

        response = requests.post(n8n_webhook_url, json=payload)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Email sent successfully!")
            return True
                
    except Exception as e:
        print(f"Error with format: {str(e)}")
    
    print("Attempt to send email failed. ")
    return False


def handle_webhook(user_input):
    cats, _ = map_user_to_geoapify(user_input)
    lat, lon = geocode_location(user_input.get("destination"))

    if lat is None or lon is None:
        print("Could not geocode destination")
        return

    places = query_geoapify_places(cats, lat, lon)

    plan = generate_email_draft(
        user_input.get("name", "Traveler"),
        user_input.get("destination", "your destination"),
        user_input.get("departure_date", "your departure date"),
        user_input.get("return_date", "your return date"),
        places
    )

    send_email_to_n8n(
        user_input.get("email", "noemail@example.com"),
        subject=f"Your Travel Plan for {user_input.get('destination', '')}",
        body=plan
    )
