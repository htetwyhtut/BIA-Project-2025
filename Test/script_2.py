import pandas as pd
import requests
import threading
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

# Set up logging to track errors and progress
logging.basicConfig(filename='places_fetch.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler
file_handler = logging.FileHandler('places_fetch.log')
# Console handler
console_handler = logging.StreamHandler()

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Google Places API Key (replace with your actual key)
API_KEY = "AIzaSyDUubz1oIkRHXOZqf-B_-6rd8DD18docvA"

# Parse command-line arguments for user configurability
parser = argparse.ArgumentParser(description='Fetch places data from Google Places API')
parser.add_argument('--radius', type=int, default=3000, help='Search radius in meters')
parser.add_argument('--max_results', type=int, default=100, help='Maximum number of results per search')
parser.add_argument('--sleep', type=float, default=2, help='Sleep time between API calls in seconds')
parser.add_argument('--min_rating', type=float, default=3.5, help='Minimum rating for places')
parser.add_argument('--min_reviews', type=int, default=10, help='Minimum number of reviews for places')
parser.add_argument('--top_n', type=int, default=100, help='Number of top places to select per district-type')
args = parser.parse_args()

# Load district and type data from CSV files
districts_df = pd.read_csv("bangkok_districts.csv")
types_df = pd.read_csv("location_types.csv")

# Progress tracking variables
total_districts = len(districts_df)
total_types = len(types_df)
total_tasks = total_districts * total_types
completed_tasks = 0
progress_lock = threading.Lock()

# Cache to store place details and avoid redundant API calls
details_cache = {}

# Helper function to make API requests with retry logic
def _make_api_request(url, params, max_retries=3, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response_json = response.json()
            status = response_json.get("status")
            if status in ["OK", "ZERO_RESULTS"]:
                return response_json
            elif status in ["OVER_QUERY_LIMIT", "UNKNOWN_ERROR"]:
                logger.warning(f"Retryable status {status} for {url}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"Non-retryable status {status} for {url}: {response_json.get('error_message')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request exception for {url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    logger.error(f"Max retries exceeded for {url}")
    return None

# Function to fetch places using the nearbysearch endpoint with retry logic
def get_places(lat, lng, keyword, radius, max_results):
    places = []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": keyword,
        "key": API_KEY
    }
    while True:
        response_json = _make_api_request(url, params)
        if response_json is None:
            break
        results = response_json.get("results", [])
        places.extend(results)
        if "next_page_token" in response_json and len(places) < max_results:
            time.sleep(args.sleep)  # Sleep to handle pagination
            params = {
                "pagetoken": response_json["next_page_token"],
                "key": API_KEY
            }
        else:
            break
    return places[:max_results]

# Function to fetch place details with caching and retry logic
def get_place_details(place_id):
    if place_id in details_cache:
        logger.info(f"Cache hit for place_id: {place_id}")
        return details_cache[place_id]
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "price_level,rating,user_ratings_total,reviews,name,types,geometry",
        "key": API_KEY
    }
    response_json = _make_api_request(url, params)
    if response_json and response_json.get("status") == "OK":
        details = response_json.get("result", {})
        details_cache[place_id] = details  # Store in cache
        return details
    else:
        logger.error(f"Failed to get details for place_id: {place_id}")
        return {}

# Function to estimate maximum cost based on price level
def estimate_dummy_cost(price_level):
    # Dummy cost logic based on Google price_level scale (0–4)
    dummy_mapping = {
        0: 500,  # Free
        1: 1000,
        2: 2000,
        3: 3000,
        4: 5000
    }
    return dummy_mapping.get(price_level, 500)  # Default to 500 if price_level is None or invalid

# Function to process a single district and its associated types
def process_district(d_row):
    global completed_tasks  # Access the global progress counter
    
    district = d_row["District"]
    lat, lng = d_row["Latitude"], d_row["Longitude"]
    
    for _, t_row in types_df.iterrows():
        interest = t_row["location_type"]
        logger.info(f"Processing {district} - {interest}")
        
        # Fetch places for the district-type combination
        places = get_places(lat, lng, interest, args.radius, args.max_results)
        
        # Filter places based on minimum rating and review count
        filtered_places = [
            p for p in places 
            if p.get("rating", 0) >= args.min_rating and p.get("user_ratings_total", 0) >= args.min_reviews
        ]
        
        # Sort and select top N places
        sorted_places = sorted(filtered_places, key=lambda x: x.get("rating", 0), reverse=True)
        top_places = sorted_places[:args.top_n]
        
        # Process each top place
        for place in top_places:
            place_id = place.get("place_id")
            if not place_id:
                logger.warning(f"No place_id for place in {district} - {interest}")
                continue
            
            # Fetch detailed information
            details = get_place_details(place_id)
            if not details:
                continue
            
            # Extract required fields
            reviews = details.get("reviews", [])[:5]  # Limit to 5 reviews
            review_texts = [r.get("text", "") for r in reviews]
            rating = details.get("rating", None)
            user_ratings = details.get("user_ratings_total", None)
            price_level = details.get("price_level", None)
            max_cost = estimate_dummy_cost(price_level)
            name = details.get("name", "")
            types = details.get("types", [])
            lat_lng = details.get("geometry", {}).get("location", {})
            latitude = lat_lng.get("lat")
            longitude = lat_lng.get("lng")
            
            # Data validation
            if (not name or 
                latitude is None or 
                longitude is None or 
                not (-90 <= latitude <= 90) or 
                not (-180 <= longitude <= 180)):
                logger.warning(f"Invalid data for place_id {place_id} in {district} - {interest}")
                continue
            
            # Append to data list
            data.append({
                "place_id": place_id,
                "name": name,
                "district": district,
                "interest": interest,
                "rating": rating,
                "user_ratings_total": user_ratings,
                "price_level": price_level,
                "max_cost": max_cost,
                "latitude": latitude,
                "longitude": longitude,
                "types": ";".join(types),
                "reviews": " ||| ".join(review_texts)
            })
            
            # Sleep to avoid hitting API rate limits
            time.sleep(args.sleep)
            
        # Update progress counter
        with progress_lock:
            completed_tasks += 1
            progress_percent = (completed_tasks / total_tasks) * 100
            logger.info(
                f"📊 Progress: {progress_percent:.1f}% complete "
                f"({completed_tasks}/{total_tasks} district-type combinations)"
            )

# Initialize data list
data = []

# Start processing
logger.info(f"🏁 Starting data collection for {total_tasks} district-type combinations")

# Process districts in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_district, [row for _, row in districts_df.iterrows()])

# Create DataFrame, drop duplicates, and save to CSV
df = pd.DataFrame(data)
df.drop_duplicates(subset=["place_id", "district", "interest"], inplace=True)
df.to_csv("training_dataset_4.csv", index=False)
logger.info("✅ Data collection completed and saved to training_dataset.csv")