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
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .weather-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.75rem 0;
    }
    .suggestion-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
    }
    .date-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.2rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .logo-container img {
        width: 120px;
        height: 120px;
        object-fit: cover;
        border-radius: 50%;
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
    st.markdown(f"**{plan.summary}**")

    risk_indicator = get_risk_color(plan.overall_risk)
    st.markdown(
        f"{risk_indicator} Risk: **{plan.overall_risk.value.upper()}** — {plan.risk_explanation}"
    )

    if multi_day:
        grouped = _group_steps_by_date(plan.steps)
        for date_key in sorted(grouped.keys()):
            if date_key == "Unscheduled":
                st.markdown('<p class="date-header">Unscheduled</p>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<p class="date-header">{format_date_human(date_key)}</p>',
                    unsafe_allow_html=True,
                )
            for step in grouped[date_key]:
                _render_step(step)
    else:
        for step in plan.steps:
            _render_step(step)


def _render_step(step):
    """Render a single TaskStep."""
    time_str = _format_time_range(step.time_from, step.time_to)
    loc_str = f" — {step.location}" if step.location else ""
    st.markdown(f"**{step.order}.** {step.description}{time_str}{loc_str}")
    if step.risk_note:
        st.caption(f"    Note: {step.risk_note}")


def display_weather_info(weather):
    """Compact weather info box."""
    sim_note = "<br><em>Simulated data</em>" if weather.is_simulated else ""
    st.markdown(
        f"""
    <div class="weather-box">
        <strong>Weather — {weather.location}</strong> &nbsp;|&nbsp;
        {format_date_human(weather.forecast_date)}<br>
        {weather.condition.title()} &nbsp;|&nbsp; {weather.temperature_celsius} °C &nbsp;|&nbsp;
        Rain {weather.precipitation_chance}% &nbsp;|&nbsp; Wind {weather.wind_speed_kmh} km/h{sim_note}
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

# Logo
_logo_path = Path(__file__).parent / "assets" / "image.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    st.markdown(
        f'<div class="logo-container"><img src="data:image/png;base64,{_logo_b64}" alt="Chronos logo"></div>',
        unsafe_allow_html=True,
    )

st.markdown('<p class="main-header">Chronos</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Weather-Adaptive Planning Agent</p>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### What are you planning?")

user_input = st.text_area(
    "Describe your plan",
    value=st.session_state.task_input,
    placeholder="e.g., Plan a beach day with friends, organize a hiking trip, arrange a garden party…",
    height=90,
    label_visibility="collapsed",
    key="task_input_widget",
)

# ── Location: city / state / country + auto-detect ────────────────────────

st.markdown("### Location")

# Auto-detect button — only runs once, caches result
detect_col, spacer_col = st.columns([1, 3])
with detect_col:
    if st.button("Detect my location"):
        with st.spinner("Detecting…"):
            detected = get_location_from_ip()
        if detected:
            st.session_state.ip_location = detected
            st.session_state.ip_location_used = False
        else:
            st.session_state.ip_location = None
            st.warning("Could not detect location. Please enter it manually.")

# Show detected location and ask for confirmation
if st.session_state.ip_location and not st.session_state.ip_location_used:
    st.info(f"Detected location: **{st.session_state.ip_location}**")
    confirm_col, reject_col, _ = st.columns([1, 1, 3])
    with confirm_col:
        if st.button("Use this location"):
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
        if st.button("Enter manually"):
            st.session_state.ip_location = None
            st.session_state.ip_location_used = False
            st.rerun()

city_col, state_col, country_col = st.columns(3)

with city_col:
    location_city = st.text_input(
        "City",
        placeholder="e.g., Mumbai",
        key="city_widget",
    )
with state_col:
    location_state = st.text_input(
        "State / Region",
        placeholder="e.g., Maharashtra",
        key="state_widget",
    )
with country_col:
    location_country = st.text_input(
        "Country",
        placeholder="e.g., India",
        key="country_widget",
    )

# ── Date range ────────────────────────────────────────────────────────────

st.markdown("### Dates")
date_col1, date_col2 = st.columns(2)

with date_col1:
    start_date = st.date_input(
        "Start Date",
        min_value=datetime.now().date(),
        key="start_date_widget",
    )
with date_col2:
    end_date = st.date_input(
        "End Date",
        min_value=datetime.now().date(),
        key="end_date_widget",
    )

# ── Generate button ───────────────────────────────────────────────────────

st.markdown("")  # spacing
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    generate_clicked = st.button(
        "Generate Plan",
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
        st.warning("Please describe what you're planning.")
        st.stop()

    location_str = _build_location_string(location_city, location_state, location_country)
    if not location_str:
        st.warning("Please enter at least a city or country, or use 'Detect my location'.")
        st.stop()

    if end_date < start_date:
        st.warning("End date cannot be before start date.")
        st.stop()

    with st.spinner("Analyzing your plan and checking weather conditions…"):
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
    st.markdown("---")

    if isinstance(response, AgentError):
        st.error(f"**Error:** {response.message}")
        st.info(f"**Suggestion:** {response.suggestion}")

    elif isinstance(response, ChronosResponse):
        is_multi_day = (
            response.start_date
            and response.end_date
            and response.start_date != response.end_date
        )

        # ── Summary row ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Location", response.extracted_location or "—")
        with col2:
            if is_multi_day:
                date_label = (
                    f"{format_date_human(response.start_date)} – "
                    f"{format_date_human(response.end_date)}"
                )
            elif response.start_date:
                date_label = format_date_human(response.start_date)
            else:
                date_label = "—"
            st.metric("Dates", date_label)
        with col3:
            if response.weather_relevance:
                relevance_text = (
                    "Yes" if response.weather_relevance.is_relevant else "No"
                )
            else:
                relevance_text = "—"
            st.metric("Weather Relevant", relevance_text)

        # ── Feasibility gate ──────────────────────────────────────────────
        if not response.task_feasibility.feasible:
            st.error(f"**Not feasible:** {response.task_feasibility.reason}")
            if response.task_feasibility.suggestion:
                st.info(f"**Suggestion:** {response.task_feasibility.suggestion}")
        else:
            # Weather
            if response.weather_data:
                display_weather_info(response.weather_data)

            st.markdown("---")

            # ── Main Plan (Plan A) ────────────────────────────────────────
            if response.plan_a:
                st.markdown("## Your Plan")
                display_plan(response.plan_a, multi_day=is_multi_day)

            # ── Suggestions by Chronos (Plan B) ──────────────────────────
            if response.plan_b:
                st.markdown("---")
                st.markdown("## Suggestions by Chronos")
                st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                st.markdown(f"**{response.plan_b.name}**")
                display_plan(response.plan_b, multi_day=is_multi_day)
                st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Previous Plans
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.saved_plans:
    st.markdown("---")
    st.markdown("## Previous Plans")
    for idx, snap in enumerate(st.session_state.saved_plans):
        label = f"{snap['request'][:60]}  —  {snap['location']}  ({snap['dates']})"
        with st.expander(label, expanded=False):
            prev = snap["response"]
            if isinstance(prev, ChronosResponse):
                prev_multi = (
                    prev.start_date
                    and prev.end_date
                    and prev.start_date != prev.end_date
                )
                if prev.plan_a:
                    st.markdown("### Your Plan")
                    display_plan(prev.plan_a, multi_day=prev_multi)
                if prev.plan_b:
                    st.markdown("### Suggestions by Chronos")
                    display_plan(prev.plan_b, multi_day=prev_multi)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#999;font-size:0.8rem;">'
    "Chronos — Weather-Adaptive Planning Agent"
    "</div>",
    unsafe_allow_html=True,
)
