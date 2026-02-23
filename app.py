"""
app.py - Streamlit UI for Chronos weather-adaptive planning agent.

Provides:
- Task input with structured location (city, state, country)
- IP-based location auto-detect with explicit user confirmation
- Date range selection (single or multi-day)
- Multi-day output grouped by date
- Saved plans history
"""

import asyncio
import base64
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

from agent import run_chronos
from models import ChronosResponse, AgentError, RiskLevel, PlanOption
from utils import get_risk_color, format_date_human, get_location_from_ip


# ──────────────────────────────────────────────────────────────────────────────
# Async helper — persistent loop that never closes
# ──────────────────────────────────────────────────────────────────────────────

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a daemon thread.

    The loop is created once and reused for every call, so libraries
    that cache loop references (httpx, pydantic_ai) never see a closed loop.
    """
    global _LOOP, _LOOP_THREAD
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()

        def _run_forever(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _LOOP_THREAD = threading.Thread(target=_run_forever, args=(_LOOP,), daemon=True)
        _LOOP_THREAD.start()
    return _LOOP


def _run_async(coro):
    """Submit an async coroutine to the persistent background loop and wait."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()  # blocks until done


# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chronos - Weather-Adaptive Planning",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global styles and variables */
    :root {
        --primary-color: #6366f1;
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --danger-gradient: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%);
        --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
        --card-shadow-hover: 0 20px 40px rgba(0,0,0,0.15);
        --border-radius: 16px;
        --text-primary: #1a202c;
        --text-secondary: #718096;
        --background: #f7fafc;
    }
    
    /* Hide Streamlit default elements */
    .css-1d391kg, .css-1v3fvcr, .css-18e3th9, .css-1dp5vir, .css-uf99v8 {
        display: none;
    }
    
    /* Main container styling */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem 1rem;
    }
    
    /* App container */
    .app-container {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        margin: 0 auto;
        max-width: 1200px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Header styles */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
        animation: fadeInScale 1s ease-out;
    }
    
    .logo-container img {
        width: 140px;
        height: 140px;
        object-fit: cover;
        border-radius: 50%;
        border: 4px solid transparent;
        background: var(--primary-gradient);
        padding: 4px;
        transition: transform 0.3s ease;
    }
    
    .logo-container img:hover {
        transform: scale(1.05) rotate(5deg);
    }
    
    /* Section headers */
    h3 {
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary);
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    /* Input cards */
    .input-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .input-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Weather box - enhanced */
    .weather-box {
        background: var(--success-gradient);
        color: white;
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
        animation: slideInUp 0.6s ease-out;
    }
    
    .weather-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    .weather-box strong {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Suggestion box - enhanced */
    .suggestion-box {
        background: linear-gradient(135deg, #f093fb 10%, #f5576c 100%);
        color: white;
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        animation: slideInRight 0.6s ease-out;
    }
    
    .suggestion-box::after {
        content: '✨';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    
    /* Date headers */
    .date-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Plan cards */
    .plan-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .plan-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Risk indicators */
    .risk-low { color: #10b981; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-high { color: #ef4444; font-weight: 600; }
    
    /* Task steps styling */
    .task-step {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .task-step:hover {
        background: #f1f5f9;
        transform: translateX(4px);
    }
    
    /* Buttons Enhancement */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Date input styling */
    .stDateInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Metrics enhancement */
    .metric-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.4s ease-out;
    }
    
    .metric-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Animations */
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(5deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: var(--border-radius);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .app-container {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .logo-container img {
            width: 100px;
            height: 100px;
        }
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div > div {
        border-color: var(--primary-color) !important;
    }
    
    /* Success/Warning/Error message styling */
    .stAlert {
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--card-shadow) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Column spacing */
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session State — persist every input and all results
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "response": None,
    "task_input": "",
    "location_city": "",
    "location_state": "",
    "location_country": "",
    # Widget keys — Streamlit reads value from these directly
    "city_widget": "",
    "state_widget": "",
    "country_widget": "",
    "start_date_widget": datetime.now().date() + timedelta(days=1),
    "end_date_widget": datetime.now().date() + timedelta(days=1),
    "saved_plans": [],          # list[dict] — snapshots of past results
    "ip_location": None,        # str | None — cached IP detection result
    "ip_location_used": False,  # whether the user accepted the detected location
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_time_range(time_from: Optional[str], time_to: Optional[str]) -> str:
    """Format ISO 8601 from/to into '8:00 AM – 10:00 AM'."""
    if not time_from and not time_to:
        return ""
    try:
        parts: list[str] = []
        for raw in (time_from, time_to):
            if raw:
                dt = datetime.fromisoformat(raw)
                parts.append(dt.strftime("%I:%M %p").lstrip("0"))
            else:
                parts.append("?")
        return f" ({parts[0]} – {parts[1]})"
    except (ValueError, TypeError):
        return f" ({time_from} – {time_to})"


def _build_location_string(city: str, state: str, country: str) -> str:
    """Combine city/state/country into a single location string."""
    parts = [p.strip() for p in (city, state, country) if p and p.strip()]
    return ", ".join(parts)


def _extract_date_from_iso(iso_str: Optional[str]) -> Optional[str]:
    """Extract YYYY-MM-DD from an ISO datetime string like '2025-07-10T09:00'."""
    if not iso_str:
        return None
    return iso_str[:10] if len(iso_str) >= 10 else None


def _group_steps_by_date(steps: list) -> dict[str, list]:
    """Group TaskStep objects by their date (from time_from)."""
    grouped: dict[str, list] = defaultdict(list)
    for step in steps:
        date_key = _extract_date_from_iso(step.time_from) or "Unscheduled"
        grouped[date_key].append(step)
    return dict(grouped)


def display_plan(plan: PlanOption, multi_day: bool = False):
    """Render a plan's steps, grouped by date when multi-day."""
    st.markdown('<div class="plan-card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 {plan.summary}")

    # Enhanced risk indicator with icons
    risk_icons = {
        "LOW": "🟢",
        "MEDIUM": "🟡", 
        "HIGH": "🔴"
    }
    risk_level = plan.overall_risk.value.upper()
    risk_icon = risk_icons.get(risk_level, "⚪")
    risk_class = f"risk-{plan.overall_risk.value.lower()}"
    
    st.markdown(
        f'<div class="{risk_class}">{risk_icon} Risk Level: <strong>{risk_level}</strong></div>',
        unsafe_allow_html=True
    )
    st.markdown(f'<p style="color: #718096; margin-top: 0.5rem;">{plan.risk_explanation}</p>', unsafe_allow_html=True)

    if multi_day:
        grouped = _group_steps_by_date(plan.steps)
        for date_key in sorted(grouped.keys()):
            if date_key == "Unscheduled":
                st.markdown('<p class="date-header">📌 Unscheduled Tasks</p>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<p class="date-header">📅 {format_date_human(date_key)}</p>',
                    unsafe_allow_html=True,
                )
            for step in grouped[date_key]:
                _render_step(step)
    else:
        for step in plan.steps:
            _render_step(step)
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_step(step):
    """Render a single TaskStep with enhanced styling."""
    time_str = _format_time_range(step.time_from, step.time_to)
    loc_str = f" 📍 {step.location}" if step.location else ""
    
    st.markdown(
        f'<div class="task-step">'
        f'<strong>⭐ Step {step.order}:</strong> {step.description}'
        f'<br><small style="color: #718096;">🕒 {time_str}{loc_str}</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    if step.risk_note:
        st.markdown(
            f'<div style="background: #fef3cd; border-left: 4px solid #f59e0b; padding: 0.5rem 1rem; '
            f'margin: 0.25rem 0 0.75rem 0; border-radius: 8px; font-size: 0.85rem;">'
            f'⚠️ <strong>Note:</strong> {step.risk_note}'
            f'</div>',
            unsafe_allow_html=True
        )


def display_weather_info(weather):
    """Enhanced weather info box with icons and better styling."""
    # Weather condition icons
    weather_icons = {
        'clear': '☀️',
        'sunny': '☀️',
        'partly cloudy': '⛅',
        'cloudy': '☁️',
        'overcast': '☁️',
        'rain': '🌧️',
        'light rain': '🌦️',
        'heavy rain': '🌧️',
        'snow': '❄️',
        'storm': '⛈️',
        'fog': '🌫️',
        'mist': '🌫️'
    }
    
    condition_lower = weather.condition.lower()
    weather_icon = '🌤️'  # default
    for key, icon in weather_icons.items():
        if key in condition_lower:
            weather_icon = icon
            break
    
    sim_badge = '<span style="background: rgba(245, 158, 11, 0.2); color: #d97706; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; margin-left: 0.5rem;">🔮 Simulated</span>' if weather.is_simulated else ''
    
    st.markdown(
        f"""
    <div class="weather-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <strong style="font-size: 1.2rem;">{weather_icon} Weather Forecast</strong>
            {sim_badge}
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; opacity: 0.8;">📍 Location</div>
                <div style="font-weight: 600;">{weather.location}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; opacity: 0.8;">📅 Date</div>
                <div style="font-weight: 600;">{format_date_human(weather.forecast_date)}</div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 1rem; text-align: center;">
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">☁️ Condition</div>
                <div style="font-weight: 600;">{weather.condition.title()}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">🌡️ Temperature</div>
                <div style="font-weight: 600;">{weather.temperature_celsius}°C</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">🌧️ Rain Chance</div>
                <div style="font-weight: 600;">{weather.precipitation_chance}%</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">💨 Wind Speed</div>
                <div style="font-weight: 600;">{weather.wind_speed_kmh} km/h</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _save_plan(response: ChronosResponse):
    """Snapshot the current response into saved_plans."""
    snapshot = {
        "request": response.original_request,
        "location": response.extracted_location or response.location_used or "—",
        "dates": f"{response.start_date or '?'} – {response.end_date or '?'}",
        "generated_at": response.generated_at,
        "response": response,
    }
    st.session_state.saved_plans.insert(0, snapshot)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

# Logo and Header - wrapped in app container
_logo_path = Path(__file__).parent / "assets" / "image.png"
logo_html = ""
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    logo_html = f'<div class="logo-container"><img src="data:image/png;base64,{_logo_b64}" alt="Chronos logo"></div>'

st.markdown(
    f'''
    <div class="app-container">
        {logo_html}
        <p class="main-header">Chronos</p>
        <p class="sub-header">🌤️ Your Weather-Adaptive Planning Assistant</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────────────────────────────────────

#st.markdown('<div class="app-container" style="margin-top: 1rem;"><div class="input-card">', unsafe_allow_html=True)
st.markdown("### 📝 What are you planning?")

user_input = st.text_area(
    "Describe your plan",
    value=st.session_state.task_input,
    placeholder="✨ e.g., Plan a beach day with friends, organize a hiking trip, arrange a garden party…",
    height=90,
    label_visibility="collapsed",
    key="task_input_widget",
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Location: city / state / country + auto-detect ────────────────────────

#st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("### 📍 Location")

# Auto-detect button — only runs once, caches result
detect_col, spacer_col = st.columns([1, 3])
with detect_col:
    if st.button("🎯 Detect my location", type="secondary"):
        with st.spinner("🔍 Detecting your location…"):
            detected = get_location_from_ip()
        if detected:
            st.session_state.ip_location = detected
            st.session_state.ip_location_used = False
        else:
            st.session_state.ip_location = None
            st.warning("🌐 Could not detect location. Please enter it manually.")

# Show detected location and ask for confirmation
if st.session_state.ip_location and not st.session_state.ip_location_used:
    st.info(f"📍 **Detected location:** {st.session_state.ip_location}")
    confirm_col, reject_col, _ = st.columns([1, 1, 3])
    with confirm_col:
        if st.button("✅ Use this location"):
            parts = [p.strip() for p in st.session_state.ip_location.split(",")]
            city = parts[0] if len(parts) >= 1 else ""
            state = parts[1] if len(parts) >= 2 else ""
            country = parts[2] if len(parts) >= 3 else ""
            # Set both canonical and widget keys so inputs update
            st.session_state.location_city = city
            st.session_state.location_state = state
            st.session_state.location_country = country
            st.session_state.city_widget = city
            st.session_state.state_widget = state
            st.session_state.country_widget = country
            st.session_state.ip_location_used = True
            st.rerun()
    with reject_col:
        if st.button("✏️ Enter manually"):
            st.session_state.ip_location = None
            st.session_state.ip_location_used = False
            st.rerun()

city_col, state_col, country_col = st.columns(3)

with city_col:
    location_city = st.text_input(
        "🏙️ City",
        placeholder="e.g., Mumbai",
        key="city_widget",
    )
with state_col:
    location_state = st.text_input(
        "🏛️ State / Region",
        placeholder="e.g., Maharashtra",
        key="state_widget",
    )
with country_col:
    location_country = st.text_input(
        "🌍 Country",
        placeholder="e.g., India",
        key="country_widget",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Date range ────────────────────────────────────────────────────────────

#st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("### 📅 Dates")
date_col1, date_col2 = st.columns(2)

with date_col1:
    start_date = st.date_input(
        "🗓️ Start Date",
        min_value=datetime.now().date(),
        key="start_date_widget",
    )
with date_col2:
    end_date = st.date_input(
        "🏁 End Date",
        min_value=datetime.now().date(),
        key="end_date_widget",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Generate button ───────────────────────────────────────────────────────

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    generate_clicked = st.button(
        "🚀 Generate Smart Plan",
        type="primary",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation & Execution
# ──────────────────────────────────────────────────────────────────────────────

if generate_clicked:
    # Clear previous result so stale errors don't linger during the new run
    st.session_state.response = None

    # Persist inputs
    st.session_state.task_input = user_input
    st.session_state.location_city = location_city
    st.session_state.location_state = location_state
    st.session_state.location_country = location_country

    if not user_input or not user_input.strip():
        st.warning("📝 Please describe what you're planning.")
        st.stop()

    location_str = _build_location_string(location_city, location_state, location_country)
    if not location_str:
        st.warning("📍 Please enter at least a city or country, or use 'Detect my location'.")
        st.stop()

    if end_date < start_date:
        st.warning("📅 End date cannot be before start date.")
        st.stop()

    with st.spinner("🤖 Analyzing your plan and checking weather conditions…"):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            response = _run_async(
                run_chronos(
                    user_request=user_input.strip(),
                    location=location_str,
                    start_date=start_str,
                    end_date=end_str,
                )
            )
            st.session_state.response = response

            # Auto-save on success
            if isinstance(response, ChronosResponse):
                _save_plan(response)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.response = AgentError(
                error_type="UnexpectedError",
                message=str(e),
                fallback_available=False,
                suggestion="Please try again or simplify your request.",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.response:
    response = st.session_state.response
    st.markdown('<div style="margin: 2rem 0; height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 1px;"></div>', unsafe_allow_html=True)

    if isinstance(response, AgentError):
        st.markdown(
            f'<div style="background: #fee2e2; color: #dc2626; padding: 1.5rem; border-radius: 16px; border-left: 4px solid #ef4444; margin: 1rem 0;">'
            f'<strong>❌ Error:</strong> {response.message}<br><br>'
            f'<strong>💡 Suggestion:</strong> {response.suggestion}'
            f'</div>',
            unsafe_allow_html=True
        )

    elif isinstance(response, ChronosResponse):
        is_multi_day = (
            response.start_date
            and response.end_date
            and response.start_date != response.end_date
        )

        # ── Enhanced Summary Metrics ───────────────────────────────────────
        st.markdown("## 📊 Plan Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">📍</div>'
                f'<div style="font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">Location</div>'
                f'<div style="font-weight: 600; color: #1a202c;">{response.extracted_location or "—"}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            if is_multi_day:
                date_label = (
                    f"{format_date_human(response.start_date)} –<br>"
                    f"{format_date_human(response.end_date)}"
                )
            elif response.start_date:
                date_label = format_date_human(response.start_date)
            else:
                date_label = "—"
            
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">📅</div>'
                f'<div style="font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">{"Duration" if is_multi_day else "Date"}</div>'
                f'<div style="font-weight: 600; color: #1a202c;">{date_label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col3:
            if response.weather_relevance:
                relevance_icon = "🌤️" if response.weather_relevance.is_relevant else "🏢"
                relevance_text = "Weather Sensitive" if response.weather_relevance.is_relevant else "Indoor/Flexible"
            else:
                relevance_icon = "❓"
                relevance_text = "Unknown"
            
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{relevance_icon}</div>'
                f'<div style="font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">Activity Type</div>'
                f'<div style="font-weight: 600; color: #1a202c;">{relevance_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Feasibility gate ──────────────────────────────────────────────
        if not response.task_feasibility.feasible:
            st.markdown(
                f'<div style="background: #fee2e2; color: #dc2626; padding: 2rem; border-radius: 16px; border-left: 4px solid #ef4444; margin: 2rem 0;">'
                f'<h3 style="color: #dc2626; margin-top: 0;">🚫 Plan Not Feasible</h3>'
                f'<p style="margin-bottom: 1rem;"><strong>Reason:</strong> {response.task_feasibility.reason}</p>',
                unsafe_allow_html=True
            )
            if response.task_feasibility.suggestion:
                st.markdown(
                    f'<p><strong>💡 Alternative Suggestion:</strong> {response.task_feasibility.suggestion}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Weather
            if response.weather_data:
                display_weather_info(response.weather_data)

            # ── Main Plan (Plan A) ────────────────────────────────────────
            if response.plan_a:
                st.markdown("## 📋 Your Original Plan")
                display_plan(response.plan_a, multi_day=is_multi_day)

            # ── Suggestions by Chronos (Plan B) ──────────────────────────
            if response.plan_b:
                st.markdown("## ✨ Chronos Smart Suggestions")
                st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 {response.plan_b.name}")
                display_plan(response.plan_b, multi_day=is_multi_day)
                st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Previous Plans
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.saved_plans:
    st.markdown('<div style="margin: 3rem 0 2rem 0; height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 1px;"></div>', unsafe_allow_html=True)
    st.markdown("## 📚 Previous Plans")
    st.markdown('<p style="color: #718096; margin-bottom: 1.5rem;">Review your previously generated plans</p>', unsafe_allow_html=True)
    
    for idx, snap in enumerate(st.session_state.saved_plans):
        # Create a more attractive label with icons and better formatting
        request_preview = snap['request'][:50] + "..." if len(snap['request']) > 50 else snap['request']
        label = f"🗂️ {request_preview} • 📍 {snap['location']} • 📅 {snap['dates']}"
        
        with st.expander(label, expanded=False):
            prev = snap["response"]
            if isinstance(prev, ChronosResponse):
                prev_multi = (
                    prev.start_date
                    and prev.end_date
                    and prev.start_date != prev.end_date
                )
                
                # Show timestamp
                st.markdown(f'<small style="color: #718096;">Generated: {prev.generated_at}</small>', unsafe_allow_html=True)
                st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
                
                if prev.plan_a:
                    st.markdown("### 📋 Original Plan")
                    display_plan(prev.plan_a, multi_day=prev_multi)
                if prev.plan_b:
                    st.markdown("### ✨ Chronos Suggestions")
                    display_plan(prev.plan_b, multi_day=prev_multi)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

# Close main app container
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">'
    '🌤️ <strong>Chronos</strong> — Your intelligent weather-adaptive planning companion<br>'
    '<small style="opacity: 0.7;">Powered by AI • Built with ❤️ by Team HackIconics</small>'
    '</div>',
    unsafe_allow_html=True,
)
