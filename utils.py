"""
utils.py - Helper functions for Chronos agent.

Keeps agent.py clean by handling:
- Location ambiguity detection
- Date parsing and interpretation
- Risk scoring calculations
- Activity classification
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from models import RiskLevel, WeatherCondition


# ──────────────────────────────────────────────────────────────────────────────
# Date Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_relative_date(text: str) -> Optional[str]:
    """
    Parse relative date expressions to YYYY-MM-DD format.
    
    Handles: today, tomorrow, this weekend, next week, etc.
    Returns None if no date expression found.
    """
    text_lower = text.lower()
    today = datetime.now()
    
    # Direct matches
    if "today" in text_lower:
        return today.strftime("%Y-%m-%d")
    
    if "tomorrow" in text_lower:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    if "this weekend" in text_lower or "weekend" in text_lower:
        # Find next Saturday
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0 and today.weekday() != 5:
            days_until_saturday = 7
        saturday = today + timedelta(days=days_until_saturday)
        return saturday.strftime("%Y-%m-%d")
    
    if "next week" in text_lower:
        # Next Monday
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        next_monday = today + timedelta(days=days_until_monday)
        return next_monday.strftime("%Y-%m-%d")
    
    # Day names (e.g., "on Friday", "this Friday")
    day_names = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    for day_name, day_num in day_names.items():
        if day_name in text_lower:
            days_ahead = (day_num - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # Assume next week if today
            target = today + timedelta(days=days_ahead)
            return target.strftime("%Y-%m-%d")
    
    # Try to find explicit date (YYYY-MM-DD or MM/DD)
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(date_pattern, text)
    if match:
        return match.group(1)
    
    # Default to tomorrow if no date found but planning implies future
    planning_keywords = ["plan", "schedule", "organize", "arrange"]
    if any(kw in text_lower for kw in planning_keywords):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return None


def format_date_human(date_str: str) -> str:
    """Convert YYYY-MM-DD to human-readable format."""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return date.strftime("%A, %B %d, %Y")
    except ValueError:
        return date_str


# ──────────────────────────────────────────────────────────────────────────────
# Location Detection
# ──────────────────────────────────────────────────────────────────────────────

# Common location indicators
LOCATION_PREPOSITIONS = ["in", "at", "near", "around", "to"]

# Known city names (subset for quick matching)
COMMON_CITIES = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "seattle",
    "denver", "boston", "miami", "atlanta", "london", "paris", "tokyo",
    "sydney", "toronto", "vancouver", "berlin", "madrid", "rome", "amsterdam"
}


def extract_location(text: str) -> Optional[str]:
    """
    Extract location from natural language text.
    
    Returns the most likely location string, or None if ambiguous/not found.
    """
    text_lower = text.lower()
    
    # Check for known cities first
    for city in COMMON_CITIES:
        if city in text_lower:
            return city.title()
    
    # Look for "in/at/near [Location]" patterns
    for prep in LOCATION_PREPOSITIONS:
        pattern = rf'\b{prep}\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'
        match = re.search(pattern, text)
        if match:
            location = match.group(1)
            # Filter out common non-location words
            non_locations = {"the", "a", "an", "my", "our", "this", "that"}
            if location.lower() not in non_locations:
                return location
    
    return None


def is_location_ambiguous(location: Optional[str]) -> bool:
    """Check if the extracted location is too vague."""
    if not location:
        return True
    
    vague_terms = {"here", "there", "somewhere", "nearby", "local", "area"}
    return location.lower() in vague_terms


def get_default_location() -> str:
    """Return a sensible default location for demos."""
    return "New York"


# ──────────────────────────────────────────────────────────────────────────────
# Activity Classification
# ──────────────────────────────────────────────────────────────────────────────

# Activities that are weather-sensitive
OUTDOOR_ACTIVITIES = {
    "picnic", "hiking", "hike", "camping", "camp", "beach", "swimming", "swim",
    "bbq", "barbecue", "garden", "gardening", "cycling", "bike", "biking",
    "running", "run", "jogging", "jog", "walking", "walk", "fishing", "fish",
    "golf", "tennis", "soccer", "football", "baseball", "park", "outdoor",
    "festival", "concert", "fair", "market", "parade", "wedding", "ceremony",
    "photography", "photoshoot", "zoo", "amusement park", "theme park",
    "kayaking", "surfing", "sailing", "boating", "climbing", "skiing"
}

# Activities that are NOT weather-sensitive
INDOOR_ACTIVITIES = {
    "meeting", "movie", "cinema", "theater", "theatre", "museum", "shopping",
    "dinner", "lunch", "restaurant", "cafe", "coffee", "gym", "workout",
    "office", "work", "study", "library", "class", "lecture", "presentation",
    "interview", "doctor", "dentist", "appointment", "spa", "massage",
    "bowling", "arcade", "escape room", "concert hall", "opera"
}


def classify_activity_weather_sensitivity(text: str) -> tuple[bool, list[str]]:
    """
    Determine if the described activity is weather-sensitive.
    
    Returns: (is_sensitive, list_of_outdoor_activities_found)
    """
    text_lower = text.lower()
    
    found_outdoor = []
    found_indoor = []
    
    for activity in OUTDOOR_ACTIVITIES:
        if activity in text_lower:
            found_outdoor.append(activity)
    
    for activity in INDOOR_ACTIVITIES:
        if activity in text_lower:
            found_indoor.append(activity)
    
    # If more outdoor than indoor activities, it's weather-sensitive
    if found_outdoor and len(found_outdoor) >= len(found_indoor):
        return True, found_outdoor
    
    # If explicitly mentions "outdoor" or "outside"
    if "outdoor" in text_lower or "outside" in text_lower:
        return True, ["outdoor activity"]
    
    # Default to weather-sensitive if no clear indoor activities
    if not found_indoor and not found_outdoor:
        # Conservative: assume weather might matter
        return True, []
    
    return bool(found_outdoor), found_outdoor


# ──────────────────────────────────────────────────────────────────────────────
# Risk Scoring
# ──────────────────────────────────────────────────────────────────────────────

def calculate_weather_risk(weather: WeatherCondition) -> RiskLevel:
    """
    Calculate risk level based on weather conditions.
    
    Factors:
    - Precipitation chance
    - Wind speed
    - Severe conditions
    """
    score = 0
    
    # Precipitation impact (0-40 points)
    if weather.precipitation_chance >= 80:
        score += 40
    elif weather.precipitation_chance >= 60:
        score += 30
    elif weather.precipitation_chance >= 40:
        score += 20
    elif weather.precipitation_chance >= 20:
        score += 10
    
    # Wind impact (0-20 points)
    if weather.wind_speed_kmh >= 40:
        score += 20
    elif weather.wind_speed_kmh >= 25:
        score += 10
    elif weather.wind_speed_kmh >= 15:
        score += 5
    
    # Severe weather conditions (0-40 points)
    severe_keywords = ["thunderstorm", "storm", "heavy rain", "hail", "severe"]
    if any(kw in weather.condition.lower() for kw in severe_keywords):
        score += 40
    elif "rain" in weather.condition.lower():
        score += 15
    elif "snow" in weather.condition.lower():
        score += 20
    
    # Convert score to risk level
    if score >= 60:
        return RiskLevel.CRITICAL
    elif score >= 40:
        return RiskLevel.HIGH
    elif score >= 20:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def get_risk_color(risk: RiskLevel) -> str:
    """Get display color for risk level (for Streamlit UI)."""
    color_map = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴"
    }
    return color_map.get(risk, "⚪")


def suggest_time_shift(weather: WeatherCondition, original_hour: int) -> Optional[int]:
    """
    Suggest a time shift if weather is better at different time.
    For simplicity, this is a heuristic - real implementation would check hourly forecast.
    
    Returns suggested hour (24h format) or None if no change recommended.
    """
    # If high precipitation chance, suggest earlier time (weather often worse in afternoon)
    if weather.precipitation_chance >= 50:
        if original_hour >= 14:  # After 2 PM
            return 10  # Suggest 10 AM
        elif original_hour >= 12:  # Noon
            return 9  # Suggest 9 AM
    
    # If very hot, suggest avoiding midday
    if weather.temperature_celsius >= 32:
        if 11 <= original_hour <= 15:
            return 17  # Suggest 5 PM
    
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Text Formatting Helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_weather_summary(weather: WeatherCondition) -> str:
    """Create a human-readable weather summary."""
    return (
        f"{weather.condition.title()}, {weather.temperature_celsius}°C, "
        f"{weather.precipitation_chance}% chance of rain, "
        f"wind {weather.wind_speed_kmh} km/h"
    )


def format_risk_explanation(risk: RiskLevel, weather: WeatherCondition) -> str:
    """Generate explanation for why a risk level was assigned."""
    explanations = []
    
    if weather.precipitation_chance >= 50:
        explanations.append(f"High precipitation chance ({weather.precipitation_chance}%)")
    
    if weather.wind_speed_kmh >= 25:
        explanations.append(f"Strong winds ({weather.wind_speed_kmh} km/h)")
    
    if "rain" in weather.condition.lower() or "storm" in weather.condition.lower():
        explanations.append(f"Unfavorable conditions ({weather.condition})")
    
    if not explanations:
        if risk == RiskLevel.LOW:
            return "Weather conditions are favorable for outdoor activities."
        else:
            return "Minor weather concerns that shouldn't significantly impact plans."
    
    return " | ".join(explanations)
