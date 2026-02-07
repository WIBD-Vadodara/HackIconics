"""
agent.py - PydanticAI agent definition for Chronos.

The reasoning core that:
1. Understands user intent
2. Decides whether weather is relevant
3. Conditionally fetches weather data
4. Generates two plan options with reasoning
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()  # Load environment variables from .env file

# Set GOOGLE_API_KEY for pydantic_ai if GEMINI_API_KEY is available
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from models import (
    ChronosResponse,
    WeatherCondition,
    WeatherRelevance,
    TaskFeasibility,
    PlanOption,
    TaskStep,
    DecisionPoint,
    RiskLevel,
    AgentError
)
from tools import get_weather
from utils import (
    classify_activity_weather_sensitivity,
    calculate_weather_risk,
    format_weather_summary,
    format_risk_explanation,
    format_date_human
)


# ──────────────────────────────────────────────────────────────────────────────
# Agent Dependencies (injected context)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChronosDependencies:
    """Runtime dependencies for the agent."""
    simulation_mode: bool = False  # Use simulated weather for demos
    user_location: Optional[str] = None  # Override location if provided
    user_date: Optional[str] = None  # Override date if provided


# ──────────────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────────────

CHRONOS_SYSTEM_PROMPT = """You are Chronos, a weather-adaptive planning assistant.

Your task is to help users optimize their plans based on weather conditions.

## REALITY CHECK (MANDATORY — run BEFORE anything else):
Before planning, you MUST validate whether the requested activity is physically
possible at the given location. Examples of INFEASIBLE requests:
- "Beach day in Anand" → Anand is an inland city with no beach
- "Skiing in Mumbai" → Mumbai has no ski slopes
- "Desert safari in Goa" → Goa is coastal, not desert

If the task is NOT feasible:
1. Set task_feasibility.feasible = false
2. Clearly explain why in task_feasibility.reason
3. Suggest a safe alternative (nearest valid location) in task_feasibility.suggestion
4. Set weather_relevance.is_relevant = false
5. Do NOT generate plan_a or plan_b — set both to null
6. Do NOT hallucinate weather data or schedules

If the task IS feasible:
1. Set task_feasibility.feasible = true
2. Briefly confirm why in task_feasibility.reason (e.g., "Miami has beaches")
3. Proceed with planning normally

## LOCATION RULES (STRICT):
- The location is provided EXPLICITLY by the user via structured input
- NEVER guess, infer, or hallucinate a location
- Use ONLY the location given in the context below
- Set location_used to the exact location provided
- Set location_confidence to 1.0 (user-provided)

## DATE RANGE RULES:
- Plans may span one or more days (start_date to end_date inclusive)
- For single-day requests start_date == end_date
- ALL steps MUST have time_from and time_to within the given date range
- NEVER schedule steps outside the date range
- For multi-day plans, organize steps day-by-day in strict chronological order
- Each step's time_from date portion MUST clearly indicate which day it belongs to
- Group logically: morning activities before afternoon, Day 1 before Day 2, etc.

## Your Process (only when feasible):
1. UNDERSTAND the user's plan request
2. DETERMINE if weather is relevant (outdoor activities, travel, events)
3. If relevant, USE the weather data provided to you
4. GENERATE two plan options:
   - Plan A: Original plan with honest risk assessment
   - Plan B: Weather-optimized alternative
5. The location is always user-provided — do not override it

## Rules:
- ALWAYS explain WHY you made each decision
- NEVER ignore weather risks - be honest about them
- Provide SPECIFIC, actionable steps (not vague advice)
- If weather is bad, suggest alternatives (different time, backup venue, etc.)
- Keep explanations concise but informative
- NEVER guess or infer a location — use only the explicitly provided location
- For EVERY step, use explicit "time_from" and "time_to" fields in ISO 8601 format (YYYY-MM-DDTHH:MM)
- NEVER output a combined time string like "08:00 - 10:00" — always use separate from/to fields
- Both time_from and time_to must be provided together (either both present or both null)
- ALL step times must fall within the given start_date to end_date range

## Output Requirements:
- You MUST return ONLY valid JSON matching the schema below
- All decisions must be traceable through the decision_trace
- Risk levels must match the actual weather conditions
- The recommended plan should be the one with lower risk
- location_used: the exact location provided by the user
- location_confidence: always 1.0 (user-provided location)

## JSON Output Schema:
{
  "original_request": "string - the user's original request",
  "location_used": "string or null - the location used for weather",
  "location_confidence": 0.9,
  "task_feasibility": {
    "feasible": true,
    "reason": "string - why the activity is or is not feasible at this location",
    "suggestion": "string or null - alternative suggestion when infeasible"
  },
  "plan_a": {
    "name": "string - plan name (e.g., 'Original Plan')",
    "summary": "string - one sentence summary",
    "steps": [
      {
        "order": 1,
        "description": "string - what to do",
        "time_from": "string or null - ISO 8601 start datetime, e.g. '2026-02-08T10:00'",
        "time_to": "string or null - ISO 8601 end datetime, e.g. '2026-02-08T12:00'",
        "location": "string or null",
        "weather_sensitive": true,
        "risk_note": "string or null"
      }
    ],
    "overall_risk": "low|medium|high|critical",
    "risk_explanation": "string - why this risk level",
    "recommended": false
  },
  "plan_b": {
    "name": "string - plan name (e.g., 'Weather-Optimized')",
    "summary": "string - one sentence summary",
    "steps": [...],
    "overall_risk": "low|medium|high|critical",
    "risk_explanation": "string",
    "recommended": true
  },
  "decision_trace": [
    {
      "decision": "string - what was decided",
      "reasoning": "string - why",
      "data_used": "string or null"
    }
  ],
  "agent_confidence": 0.85
}

## When Weather is NOT Relevant:
- Indoor activities (meetings, movies, etc.)
- Virtual events
- Activities that explicitly don't care about weather

When weather is not relevant, still provide two plans but note that weather 
doesn't significantly impact either option.

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, just valid JSON.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Agent Definition
# ──────────────────────────────────────────────────────────────────────────────

# Create a single agent instance
chronos_agent = Agent(
    "google-gla:gemini-2.5-flash",
    system_prompt=CHRONOS_SYSTEM_PROMPT
)


def parse_agent_response(result_text: str) -> ChronosResponse:
    """Parse and validate the agent's JSON output into ChronosResponse."""
    # Clean up the response - remove markdown code blocks if present
    cleaned = result_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Parse JSON and validate with Pydantic
    data = json.loads(cleaned)
    return ChronosResponse.model_validate(data)


# ──────────────────────────────────────────────────────────────────────────────
# Main Planning Function
# ──────────────────────────────────────────────────────────────────────────────

async def run_chronos(
    user_request: str,
    location: str,
    start_date: str,
    end_date: str,
    simulation_mode: bool = False,
    model_name: str = "gemini-2.5-flash"
) -> ChronosResponse | AgentError:
    """
    Execute the Chronos planning flow.
    
    This is the main entry point that:
    1. Uses the explicit location provided by the user
    2. Determines weather relevance
    3. Fetches weather if needed (for each date in range)
    4. Runs the agent to generate plans
    5. Returns structured response
    
    NEVER raises exceptions - always returns a result or error object.
    """
    try:
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Use explicit inputs (no inference)
        # ─────────────────────────────────────────────────────────────────────
        
        # Location is always user-provided — no guessing
        location = location.strip()
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Determine weather relevance
        # ─────────────────────────────────────────────────────────────────────
        
        is_weather_relevant, outdoor_activities = classify_activity_weather_sensitivity(user_request)
        
        weather_relevance = WeatherRelevance(
            is_relevant=is_weather_relevant,
            confidence=0.9 if outdoor_activities else 0.7,
            explanation=(
                f"Identified outdoor activities: {', '.join(outdoor_activities)}"
                if outdoor_activities
                else "No specific outdoor activities identified, but weather may still be relevant"
            ),
            outdoor_activities=outdoor_activities
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Fetch weather (conditionally)
        # ─────────────────────────────────────────────────────────────────────
        
        weather_data: Optional[WeatherCondition] = None
        decision_trace: list[DecisionPoint] = []
        
        if is_weather_relevant:
            # Fetch weather for the start date (primary planning day)
            weather_data = await get_weather(location, start_date, use_simulation=simulation_mode)
            
            decision_trace.append(DecisionPoint(
                decision="Fetched weather data",
                reasoning=f"Weather is relevant for outdoor activities: {outdoor_activities}",
                data_used=format_weather_summary(weather_data)
            ))
        else:
            decision_trace.append(DecisionPoint(
                decision="Skipped weather lookup",
                reasoning="Activity appears to be primarily indoor or weather-independent",
                data_used=None
            ))
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Run the agent
        # ─────────────────────────────────────────────────────────────────────
        
        # Build the prompt with all context
        context_prompt = build_agent_prompt(
            user_request=user_request,
            location=location,
            start_date=start_date,
            end_date=end_date,
            weather_data=weather_data,
            weather_relevance=weather_relevance
        )
        
        # Run agent and parse the JSON response
        result = await chronos_agent.run(context_prompt)
        response = parse_agent_response(result.output)
        
        # Enrich response with pre-computed data
        response.extracted_location = location
        response.start_date = start_date
        response.end_date = end_date
        response.weather_relevance = weather_relevance
        response.weather_data = weather_data
        response.decision_trace = decision_trace + response.decision_trace
        
        # Location is always explicit — set deterministically
        response.location_used = location
        response.location_confidence = 1.0
        
        return response
        
    except Exception as e:
        # Graceful error handling - NEVER crash during demo
        return AgentError(
            error_type=type(e).__name__,
            message=str(e),
            fallback_available=True,
            suggestion="Try simplifying your request or check your API configuration."
        )


def build_agent_prompt(
    user_request: str,
    location: str,
    start_date: str,
    end_date: str,
    weather_data: Optional[WeatherCondition],
    weather_relevance: WeatherRelevance
) -> str:
    """Build the full prompt for the agent with all context."""
    
    if start_date == end_date:
        date_display = f"{format_date_human(start_date)} ({start_date})"
    else:
        date_display = f"{format_date_human(start_date)} to {format_date_human(end_date)} ({start_date} to {end_date})"
    
    prompt_parts = [
        f"## User Request\n{user_request}",
        f"\n## Provided Context (user-supplied — do NOT override)",
        f"- Location: {location}",
        f"- Date range: {date_display}",
    ]
    
    prompt_parts.append(f"\n## Weather Relevance Assessment")
    prompt_parts.append(f"- Relevant: {weather_relevance.is_relevant}")
    prompt_parts.append(f"- Confidence: {weather_relevance.confidence:.0%}")
    prompt_parts.append(f"- Outdoor activities: {weather_relevance.outdoor_activities}")
    
    if weather_data:
        risk_level = calculate_weather_risk(weather_data)
        prompt_parts.append(f"\n## Weather Data for {weather_data.location} on {weather_data.forecast_date}")
        prompt_parts.append(f"- Condition: {weather_data.condition}")
        prompt_parts.append(f"- Temperature: {weather_data.temperature_celsius}°C")
        prompt_parts.append(f"- Precipitation chance: {weather_data.precipitation_chance}%")
        prompt_parts.append(f"- Wind speed: {weather_data.wind_speed_kmh} km/h")
        prompt_parts.append(f"- Humidity: {weather_data.humidity_percent}%")
        prompt_parts.append(f"- Calculated risk level: {risk_level.value}")
        if weather_data.is_simulated:
            prompt_parts.append("- **WARNING: This is ESTIMATED weather data (forecast unavailable for this date). Do NOT present these numbers as real. Clearly tell the user the forecast is an estimate.**")
    
    prompt_parts.append("\n## Your Task")
    prompt_parts.append("FIRST: Perform the REALITY CHECK. Decide whether the activity is physically feasible at the location.")
    prompt_parts.append("Fill in 'task_feasibility' BEFORE generating any plans.")
    prompt_parts.append("If infeasible: set plan_a and plan_b to null, do NOT invent schedules or weather.")
    prompt_parts.append("If feasible: generate a ChronosResponse with two plan options.")
    prompt_parts.append("Plan A should be the original plan with honest risk assessment.")
    prompt_parts.append("Plan B should be a weather-optimized alternative.")
    prompt_parts.append("Include a decision trace explaining each key decision.")
    prompt_parts.append("IMPORTANT: For every step, use 'time_from' and 'time_to' fields in ISO 8601 format (YYYY-MM-DDTHH:MM). NEVER use a combined time string.")
    prompt_parts.append(f"IMPORTANT: ALL step times MUST fall within {start_date} to {end_date} (inclusive). Do NOT schedule outside this range.")
    prompt_parts.append("IMPORTANT: Use ONLY the location provided above. Do NOT guess or change it.")
    
    return "\n".join(prompt_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Fallback Plan Generator (for when agent fails)
# ──────────────────────────────────────────────────────────────────────────────

def generate_fallback_response(
    user_request: str,
    location: str,
    date: str,
    weather_data: Optional[WeatherCondition]
) -> ChronosResponse:
    """
    Generate a basic response when the LLM agent fails.
    This ensures we always have something to show in demos.
    """
    risk_level = (
        calculate_weather_risk(weather_data) 
        if weather_data 
        else RiskLevel.MEDIUM
    )
    
    weather_relevance = WeatherRelevance(
        is_relevant=True,
        confidence=0.5,
        explanation="Fallback mode - assuming weather may be relevant",
        outdoor_activities=[]
    )
    
    plan_a = PlanOption(
        name="Original Plan",
        summary=f"Proceed with your plan as requested",
        steps=[
            TaskStep(
                order=1,
                description=f"Proceed with: {user_request}",
                time_from=f"{date}T09:00",
                time_to=f"{date}T17:00",
                weather_sensitive=True,
                risk_note=f"Weather risk: {risk_level.value}"
            )
        ],
        overall_risk=risk_level,
        risk_explanation=format_risk_explanation(risk_level, weather_data) if weather_data else "Weather data unavailable",
        recommended=risk_level == RiskLevel.LOW
    )
    
    plan_b = PlanOption(
        name="Weather-Conscious Alternative",
        summary="Consider weather conditions when proceeding",
        steps=[
            TaskStep(
                order=1,
                description="Check weather before leaving",
                time_from=f"{date}T08:00",
                time_to=f"{date}T08:30",
                weather_sensitive=False
            ),
            TaskStep(
                order=2,
                description=f"Proceed with: {user_request}",
                time_from=f"{date}T09:00",
                time_to=f"{date}T17:00",
                weather_sensitive=True,
                risk_note="Have a backup plan ready"
            )
        ],
        overall_risk=RiskLevel.LOW if risk_level != RiskLevel.CRITICAL else RiskLevel.MEDIUM,
        risk_explanation="Taking precautions reduces risk",
        recommended=risk_level != RiskLevel.LOW
    )
    
    return ChronosResponse(
        original_request=user_request,
        extracted_location=location,
        start_date=date,
        end_date=date,
        task_feasibility=TaskFeasibility(
            feasible=True,
            reason="Fallback mode — feasibility not evaluated",
            suggestion=None
        ),
        weather_relevance=weather_relevance,
        weather_data=weather_data,
        plan_a=plan_a,
        plan_b=plan_b,
        decision_trace=[
            DecisionPoint(
                decision="Used fallback planning",
                reasoning="LLM agent unavailable, using rule-based planning",
                data_used=None
            )
        ],
        agent_confidence=0.3
    )
