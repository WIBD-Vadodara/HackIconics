"""
models.py - Structured output contracts for Chronos agent.

These Pydantic models define EXACTLY what the agent can output.
No free-form text - everything is typed and validated.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk assessment levels for weather impact."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherCondition(BaseModel):
    """Weather data structure from external API or simulation."""
    temperature_celsius: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (sunny, rainy, cloudy, etc.)")
    precipitation_chance: int = Field(ge=0, le=100, description="Precipitation probability %")
    wind_speed_kmh: float = Field(ge=0, description="Wind speed in km/h")
    humidity_percent: int = Field(ge=0, le=100, description="Humidity percentage")
    forecast_date: str = Field(description="Date of forecast (YYYY-MM-DD)")
    location: str = Field(description="Location name")
    is_simulated: bool = Field(default=False, description="Whether this is simulated data")


class TaskStep(BaseModel):
    """A single step in a plan."""
    order: int = Field(ge=1, description="Step order (1-indexed)")
    description: str = Field(description="What to do in this step")
    time_from: Optional[str] = Field(default=None, description="Start time as ISO 8601 datetime (YYYY-MM-DDTHH:MM)")
    time_to: Optional[str] = Field(default=None, description="End time as ISO 8601 datetime (YYYY-MM-DDTHH:MM)")
    location: Optional[str] = Field(default=None, description="Where this step takes place")
    weather_sensitive: bool = Field(default=False, description="Is this step affected by weather?")
    risk_note: Optional[str] = Field(default=None, description="Weather-related risk for this step")


class PlanOption(BaseModel):
    """A complete plan option with risk assessment."""
    name: str = Field(description="Plan name (e.g., 'Original Plan', 'Weather-Optimized')")
    summary: str = Field(description="One-sentence summary of this plan")
    steps: list[TaskStep] = Field(description="Ordered list of steps")
    overall_risk: RiskLevel = Field(description="Overall weather risk level")
    risk_explanation: str = Field(description="Why this risk level was assigned")
    recommended: bool = Field(default=False, description="Is this the recommended option?")


class DecisionPoint(BaseModel):
    """A single decision made by the agent with reasoning."""
    decision: str = Field(description="What was decided")
    reasoning: str = Field(description="Why this decision was made")
    data_used: Optional[str] = Field(default=None, description="What data informed this decision")


class TaskFeasibility(BaseModel):
    """Reality check: is the requested activity physically possible at the location?"""
    feasible: bool = Field(description="Whether the activity is feasible at the given location")
    reason: str = Field(description="Explanation of why the task is or is not feasible")
    suggestion: Optional[str] = Field(default=None, description="Alternative suggestion when infeasible (e.g., nearest valid location)")


class WeatherRelevance(BaseModel):
    """Agent's assessment of whether weather matters for this plan."""
    is_relevant: bool = Field(description="Does weather affect this plan?")
    confidence: float = Field(ge=0, le=1, description="Confidence in this assessment (0-1)")
    explanation: str = Field(description="Why weather is or isn't relevant")
    outdoor_activities: list[str] = Field(default_factory=list, description="Identified outdoor activities")


class ChronosResponse(BaseModel):
    """
    Complete agent response with both plan options and decision trace.
    This is the structured output the agent MUST return.
    """
    # Input understanding
    original_request: str = Field(description="The user's original request")
    extracted_location: Optional[str] = Field(default=None, description="Location extracted from request")
    start_date: Optional[str] = Field(default=None, description="Start date of the plan (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date of the plan (YYYY-MM-DD)")
    location_used: Optional[str] = Field(default=None, description="The actual location used for weather query")
    location_confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence in location accuracy (0-1)")
    
    # Feasibility check (MUST be evaluated before planning)
    task_feasibility: TaskFeasibility = Field(description="Reality check on whether the task is feasible at the location")
    
    # Weather assessment (set by code, not LLM)
    weather_relevance: Optional[WeatherRelevance] = Field(default=None, description="Assessment of weather relevance")
    weather_data: Optional[WeatherCondition] = Field(default=None, description="Weather data if fetched")
    
    # Generated plans (only meaningful when task_feasibility.feasible is true)
    plan_a: Optional[PlanOption] = Field(default=None, description="Original plan with risks noted")
    plan_b: Optional[PlanOption] = Field(default=None, description="Weather-optimized alternative")
    
    # Decision trace
    decision_trace: list[DecisionPoint] = Field(default_factory=list, description="Key decisions and their reasoning")
    
    # Meta
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent_confidence: float = Field(ge=0, le=1, description="Overall confidence in recommendations")


class AgentError(BaseModel):
    """Structured error response for graceful failures."""
    error_type: str = Field(description="Type of error encountered")
    message: str = Field(description="Human-readable error message")
    fallback_available: bool = Field(default=True, description="Can we provide a fallback?")
    suggestion: str = Field(description="What the user can try instead")
