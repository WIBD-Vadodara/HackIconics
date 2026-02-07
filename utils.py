"""
utils.py - Helper functions for Chronos agent.

Keeps agent.py clean by handling:
- Location ambiguity detection
- Date parsing and interpretation
- Risk scoring calculations
- Activity classification
"""

import re
import requests
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

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
# Location Input & Detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LocationInput:
    """Structured location input with multiple levels of specificity."""
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    auto_detect: bool = False
    
    def __str__(self) -> str:
        """Generate location string from available components."""
        parts = []
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts) if parts else None
    
    def is_empty(self) -> bool:
        """Check if no explicit location is provided."""
        return not self.city and not self.state and not self.country
    
    def confidence(self) -> float:
        """
        Calculate input confidence (0.0 to 1.0).
        City only: 1.0 (strongest signal)
        City + State/Country: 0.9
        State + Country: 0.7 (weaker signal)
        Country only: 0.5
        """
        if self.city:
            return 1.0 if (self.state or self.country) else 0.95
        elif self.state and self.country:
            return 0.7
        elif self.country:
            return 0.5
        return 0.0


def get_location_from_ip() -> Optional[str]:
    """
    Auto-detect location via IP using wttr.in API.
    
    Returns: location string (city, region, country) or None if detection fails
    """
    try:
        response = requests.get("https://wttr.in/?format=j1", timeout=5)
        if response.status_code == 200:
            data = response.json()
            nearest = data.get("nearest_area", [{}])[0]
            
            city = nearest.get("areaName", [{}])[0].get("value")
            region = nearest.get("region", [{}])[0].get("value")
            country = nearest.get("country", [{}])[0].get("value")
            
            # Build location string with available components
            parts = []
            if city:
                parts.append(city)
            if region:
                parts.append(region)
            if country:
                parts.append(country)
            
            return ", ".join(parts) if parts else None
    except (requests.RequestException, KeyError, ValueError):
        # Silently fail if IP detection doesn't work
        pass
    
    return None


def normalize_location(
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    text_input: Optional[str] = None,
    auto_detect: bool = False
) -> Optional[str]:
    """
    Normalize location from UI inputs following priority order:
    
    Priority 1 (Explicit):
    - If city provided → return immediately (strongest signal)
    - If state + country → combine them
    - If country only → return country
    
    Priority 2 (Partial):
    - Parse text_input for "City, State/Country" or "City, Country" patterns
    - Extract recognized cities
    
    Priority 3 (Implicit):
    - If auto_detect=True → use IP geolocation
    - If auto_detect=False and all empty → return None with warning
    
    Returns: formatted location string suitable for weather_tool()
    """
    
    # Priority 1: Explicit inputs with strongest signal first
    if city:
        # City is the strongest signal
        parts = [city.strip()]
        if state:
            parts.append(state.strip())
        if country:
            parts.append(country.strip())
        return ", ".join(parts)
    
    # Second strongest: state + country combination
    if state and country:
        return f"{state.strip()}, {country.strip()}"
    
    # Single country
    if country:
        return country.strip()
    
    # Priority 2: Parse text input for partial location patterns
    if text_input:
        location = extract_location_from_text(text_input)
        if location:
            return location
    
    # Priority 3: Auto-detect via IP or fail
    if auto_detect:
        detected = get_location_from_ip()
        if detected:
            return detected
        # Fall through to None if detection fails
    
    return None


def extract_location_from_text(text: str) -> Optional[str]:
    """
    Extract location from natural language text.
    
    Handles patterns like:
    - "Vadodara" (city only)
    - "Gujarat, India" (state, country)
    - "New York, USA" (city, country)
    """
    text_lower = text.lower().strip()
    
    # Check for known cities first (strongest partial signal)
    for city in COMMON_CITIES:
        if city in text_lower:
            return city.title()
    
    # Look for "City, State" or "City, Country" patterns
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= 2:
            # Assume format: [City/State], [State/Country/Country]
            return ", ".join(parts).title()
    
    # Look for "in/at/near [Location]" patterns
    location_prepositions = ["in", "at", "near", "around", "to"]
    for prep in location_prepositions:
        pattern = rf'\b{prep}\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'
        match = re.search(pattern, text)
        if match:
            location = match.group(1)
            # Filter out common non-location words
            non_locations = {"the", "a", "an", "my", "our", "this", "that"}
            if location.lower() not in non_locations:
                return location
    
    return None


# Common location indicators
LOCATION_PREPOSITIONS = ["in", "at", "near", "around", "to"]

# Known city names (subset for quick matching)
COMMON_CITIES = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "seattle",
    "denver", "boston", "miami", "atlanta", "london", "paris", "tokyo",
    "sydney", "toronto", "vancouver", "berlin", "madrid", "rome", "amsterdam",
    "vadodara", "mumbai", "delhi", "bangalore", "hyderabad", "pune", "kolkata"
}


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
