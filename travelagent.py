from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini

# =============================
# 🔑 API Key Setup
# =============================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# =============================
# ⚙️ Flask App Init
# =============================
app = Flask(__name__)
CORS(app, origins=["https://vibenavigator247.vercel.app/"]) 

# =============================
# 🧠 Agents Setup
# =============================
researcher = Agent(
    name="Researcher",
    instructions=[
        "Identify the travel destination specified by the user.",
        "Gather detailed information on the destination, including climate, culture, and safety tips.",
        "Find popular attractions, landmarks, and must-visit places.",
        "Search for activities that match the user’s interests and travel style.",
        "Prioritize information from reliable sources and official travel guides.",
        "Provide well-structured summaries with key insights and recommendations."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

hotel_restaurant_finder = Agent(
    name="Hotel & Restaurant Finder",
    instructions=[
        "Identify key locations in the user's travel itinerary.",
        "Search for highly rated hotels near those locations.",
        "Search for top-rated restaurants based on cuisine preferences and proximity.",
        "Prioritize results based on user preferences, ratings, and availability.",
        "Provide direct booking links or reservation options where possible."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

planner = Agent(
    name="Planner",
    instructions=[
        "Gather details about the user's travel preferences and budget.",
        "Create a detailed itinerary with scheduled activities and estimated costs.",
        "Ensure the itinerary includes transportation options and travel time estimates.",
        "Optimize the schedule for convenience and enjoyment.",
        "Present the itinerary in a structured format."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    add_datetime_to_instructions=True,
)

# =============================
# 🚀 Travel Planning Endpoint
# =============================
@app.route('/plan-trip', methods=['POST'])
def plan_trip():
    data = request.json

    # Extract input
    source = data.get('source')
    destination = data.get('destination')
    num_days = data.get('num_days', 5)
    travel_theme = data.get('travel_theme', "Solo Exploration")
    activity_preferences = data.get('activities', "sightseeing")
    departure_date = data.get('departure_date')
    return_date = data.get('return_date')
    budget = data.get('budget', "Standard")
    flight_class = data.get('flight_class', "Economy")
    hotel_rating = data.get('hotel_rating', "Any")
    visa_required = data.get('visa_required', False)
    travel_insurance = data.get('travel_insurance', False)

    try:
        # Step 1: Research destination
        research_prompt = (
            f"Research the best attractions and activities in {destination} for a {num_days}-day {travel_theme.lower()} trip. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. "
            f"Hotel Rating: {hotel_rating}. Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}."
        )
        research_results = researcher.run(research_prompt, stream=False)

        # Step 2: Find hotels & restaurants
        hotel_prompt = (
            f"Find the best hotels and restaurants near popular attractions in {destination} for a {travel_theme.lower()} trip. "
            f"Budget: {budget}. Hotel Rating: {hotel_rating}. Preferred activities: {activity_preferences}."
        )
        hotel_results = hotel_restaurant_finder.run(hotel_prompt, stream=False)

        # Step 3: Create itinerary
        itinerary_prompt = (
            f"Based on the following data, create a {num_days}-day itinerary for a {travel_theme.lower()} trip to {destination}. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. Hotel Rating: {hotel_rating}. "
            f"Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}. Research: {research_results.content}. "
            f"Hotels & Restaurants: {hotel_results.content}."
        )
        itinerary = planner.run(itinerary_prompt, stream=False)

        # Return all 3 outputs
        return jsonify({
            "destination": destination,
            "research_summary": research_results.content,
            "hotels_and_restaurants": hotel_results.content,
            "itinerary": itinerary.content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================
# ▶️ Run API Server
# =============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
