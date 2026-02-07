"""
app.py - Streamlit UI for Chronos weather-adaptive planning agent.

Provides:
- User input collection
- Agent execution
- Plan visualization with risk indicators
- Decision trace display
"""

import asyncio
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st

from agent import run_chronos, generate_fallback_response
from models import ChronosResponse, AgentError, RiskLevel, PlanOption
from tools import get_weather
from utils import (
    get_risk_color,
    format_date_human,
    format_weather_summary,
    get_default_location,
    normalize_location
)


# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chronos - Weather-Adaptive Planning",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .plan-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E3A5F;
    }
    .recommended-card {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .decision-trace {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #6c757d;
    }
    .weather-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────

if "response" not in st.session_state:
    st.session_state.response = None

if "history" not in st.session_state:
    st.session_state.history = []


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_risk_badge_style(risk: RiskLevel) -> tuple[str, str]:
    """Get background color and text for risk badge."""
    styles = {
        RiskLevel.LOW: ("#d4edda", "#155724"),
        RiskLevel.MEDIUM: ("#fff3cd", "#856404"),
        RiskLevel.HIGH: ("#ffe5d0", "#c65102"),
        RiskLevel.CRITICAL: ("#f8d7da", "#721c24")
    }
    return styles.get(risk, ("#e9ecef", "#495057"))


def display_plan_card(plan: PlanOption, plan_label: str):
    """Display a single plan option as a styled card."""
    bg_color, text_color = get_risk_badge_style(plan.overall_risk)
    
    # Card header
    header_emoji = "⭐ " if plan.recommended else ""
    recommended_text = " (Recommended)" if plan.recommended else ""
    
    st.markdown(f"### {header_emoji}{plan_label}: {plan.name}{recommended_text}")
    
    # Summary
    st.markdown(f"**{plan.summary}**")
    
    # Risk badge
    risk_emoji = get_risk_color(plan.overall_risk)
    st.markdown(
        f"{risk_emoji} **Risk Level:** {plan.overall_risk.value.upper()} — {plan.risk_explanation}"
    )
    
    # Steps
    st.markdown("#### Steps:")
    for step in plan.steps:
        weather_icon = "🌤️" if step.weather_sensitive else "🏠"
        time_str = f" ({step.time_suggestion})" if step.time_suggestion else ""
        location_str = f" @ {step.location}" if step.location else ""
        
        st.markdown(f"""
        {step.order}. {weather_icon} **{step.description}**{time_str}{location_str}
        """)
        
        if step.risk_note:
            st.caption(f"   ⚠️ {step.risk_note}")


def display_weather_info(weather):
    """Display weather information in a styled box."""
    st.markdown(f"""
    <div class="weather-box">
        <h4>🌡️ Weather Forecast for {weather.location}</h4>
        <p><strong>{weather.condition.title()}</strong></p>
        <p>🌡️ {weather.temperature_celsius}°C | 💧 {weather.precipitation_chance}% rain | 💨 {weather.wind_speed_kmh} km/h</p>
        <p>📅 {format_date_human(weather.forecast_date)}</p>
        {"<p><em>⚠️ Simulated data</em></p>" if weather.is_simulated else ""}
    </div>
    """, unsafe_allow_html=True)


def display_decision_trace(decisions):
    """Display the agent's decision trace."""
    st.markdown("### 🧠 Decision Trace")
    st.caption("How Chronos reasoned through your request:")
    
    for i, decision in enumerate(decisions, 1):
        with st.expander(f"Decision {i}: {decision.decision}", expanded=i == 1):
            st.markdown(f"**Reasoning:** {decision.reasoning}")
            if decision.data_used:
                st.markdown(f"**Data Used:** {decision.data_used}")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model selection
    model_name = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
        help="Select the Gemini model to use"
    )
    
    # Simulation mode
    simulation_mode = st.toggle(
        "🎭 Demo Mode",
        value=True,
        help="Use simulated weather data for reliable demos"
    )
    
    st.markdown("---")
    
    # Structured location input
    st.markdown("### 📍 Location")
    
    location_city = st.text_input(
        "City",
        placeholder="e.g., Vadodara",
        help="City name (optional)"
    )
    
    location_state = st.text_input(
        "State / Region",
        placeholder="e.g., Gujarat",
        help="State or region (optional)"
    )
    
    location_country = st.text_input(
        "Country",
        placeholder="e.g., India",
        help="Country name (optional)"
    )
    
    auto_detect_location = st.checkbox(
        "Auto-detect my location if empty",
        value=True,
        help="Use IP geolocation to detect location if no manual input is provided"
    )
    
    override_date = st.date_input(
        "Date",
        value=datetime.now() + timedelta(days=1),
        min_value=datetime.now(),
        help="Override auto-detected date"
    )
    
    st.markdown("---")
    
    # API Key status
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        st.success("✅ Gemini API key configured")
    else:
        st.warning("⚠️ No Gemini API key found")
        st.caption("Set GEMINI_API_KEY environment variable")
    
    st.markdown("---")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.response = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">⏱️ Chronos</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Weather-Adaptive Planning Agent</p>', unsafe_allow_html=True)

# Input area
st.markdown("### 📝 What are you planning?")

# Example prompts
example_prompts = [
    "Plan a picnic in Central Park this weekend",
    "Organize a beach day in Miami next Saturday",
    "Schedule an outdoor wedding ceremony in Seattle",
    "Plan a hiking trip to Denver tomorrow",
    "Arrange a garden party in London on Friday"
]

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "Describe your plan",
        placeholder="e.g., Plan a picnic in the park this Saturday afternoon...",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Try an example:**")
    for prompt in example_prompts[:3]:
        if st.button(prompt[:30] + "...", key=prompt, use_container_width=True):
            user_input = prompt
            st.session_state.example_used = prompt

# Generate button
generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
with generate_col2:
    generate_clicked = st.button(
        "🚀 Generate Optimized Plans",
        type="primary",
        use_container_width=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# Agent Execution
# ──────────────────────────────────────────────────────────────────────────────

if generate_clicked and user_input:
    with st.spinner("🤔 Chronos is analyzing your plan and checking weather conditions..."):
        # Build location from structured inputs
        try:
            location_override = normalize_location(
                city=location_city.strip() if location_city else None,
                state=location_state.strip() if location_state else None,
                country=location_country.strip() if location_country else None,
                auto_detect=auto_detect_location
            )
        except ValueError as e:
            st.error(f"Location error: {str(e)}")
            location_override = None
        
        # Check if location is required but missing
        if not location_override and not auto_detect_location:
            st.warning("⚠️ Please provide a location or enable auto-detect.")
            st.stop()
        
        date_override = override_date.strftime("%Y-%m-%d") if override_date else None
        
        # Run the agent
        try:
            response = asyncio.run(
                run_chronos(
                    user_request=user_input,
                    simulation_mode=simulation_mode,
                    override_location=location_override,
                    override_date=date_override,
                    model_name=model_name
                )
            )
            
            # Store in session state
            st.session_state.response = response
            
            # Add to history
            st.session_state.history.append({
                "request": user_input,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.session_state.response = AgentError(
                error_type="UnexpectedError",
                message=str(e),
                fallback_available=False,
                suggestion="Please try again or simplify your request."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.response:
    response = st.session_state.response
    
    st.markdown("---")
    
    if isinstance(response, AgentError):
        # Error display
        st.error(f"**Error:** {response.message}")
        st.info(f"💡 **Suggestion:** {response.suggestion}")
        
    elif isinstance(response, ChronosResponse):
        # Success display
        
        # Context summary
        st.markdown("### 📋 Request Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📍 Location", response.extracted_location or "Not specified")
        with col2:
            st.metric("📅 Date", format_date_human(response.extracted_date) if response.extracted_date else "Not specified")
        with col3:
            relevance_text = "Yes" if response.weather_relevance.is_relevant else "No"
            st.metric("🌤️ Weather Relevant", relevance_text)
        with col4:
            confidence_pct = f"{int(response.location_confidence * 100)}%"
            st.metric("📍 Location Confidence", confidence_pct)
        
        # Weather info (if fetched)
        if response.weather_data:
            display_weather_info(response.weather_data)
        
        # Show location_used if different from extracted_location
        if response.location_used and response.location_used != response.extracted_location:
            st.info(f"📍 Using location: **{response.location_used}**" + 
                   (f" (confidence: {int(response.location_confidence * 100)}%)" if response.location_confidence < 1.0 else ""))
        
        st.markdown("---")
        
        # Plan options
        st.markdown("## 📊 Your Plan Options")
        
        plan_col1, plan_col2 = st.columns(2)
        
        with plan_col1:
            display_plan_card(response.plan_a, "Option A")
        
        with plan_col2:
            display_plan_card(response.plan_b, "Option B")
        
        st.markdown("---")
        
        # Decision trace (collapsible)
        with st.expander("🧠 View Decision Trace", expanded=False):
            display_decision_trace(response.decision_trace)
        
        # Confidence indicator
        st.markdown("---")
        confidence_pct = int(response.agent_confidence * 100)
        st.progress(response.agent_confidence, text=f"Agent Confidence: {confidence_pct}%")


# ──────────────────────────────────────────────────────────────────────────────
# History (collapsible)
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.history:
    with st.expander(f"📜 History ({len(st.session_state.history)} requests)", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{i}.** {item['request'][:50]}...")
            st.caption(item['timestamp'])


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        ⏱️ Chronos — Weather-Adaptive Planning Agent<br>
        Built with PydanticAI + Streamlit | Demo Mode Active: Simulated Weather
    </div>
    """,
    unsafe_allow_html=True
)
