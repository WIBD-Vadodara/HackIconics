# WiBD GenAI Hackathon 2026

## Team Name
HackIconics

---

## Problem Statement
- Problem Statement Number: 5 (Weather Intelligence)
- Problem Statement Title: Chronos – Weather-Adaptive Planning Assistant

---

## Project Overview
> “Most apps tell you it’s raining. Chronos tells you what to do about it.” – Team HackIconics

Chronos is a weather-intelligent planning companion that fuses large-language-model reasoning with live (or simulated) meteorological data to eliminate guesswork from outdoor itineraries.
- **Challenge**: Traditional planners are static—they only alert you after weather ruins the plan. Chronos reasons *before* it acts, asking “does weather even matter?” and adapting proactively when it does.
- **Audience**: Individuals, families, logistics leads, and hackathon evaluators who need provably safe itineraries with minimal manual research.
- **Value Proposition**: The agent rejects infeasible ideas (e.g., “beach day in an inland city”), quantifies weather risk with transparent reasoning, and always offers a weather-optimized alternative so users can act with confidence.

**Key Capabilities (from PPT narrative)**
- **Temporal Tetris**: Dynamically reschedules steps to sidestep bad weather instead of simply cancelling plans.
- **Rain-Aware Buffers**: Injects travel/transition buffers when precipitation is likely, modeling “programmatic empathy.”
- **Decision Transparency**: Displays Option A vs. Option B plus a reasoning trace, so stakeholders see *why* recommendations changed.
- **Resilient Demos**: Simulation mode mirrors the real pipeline to guarantee stable judging sessions even without network access.

---

## Tech Stack
- **Programming Language(s)**: Python 3.10+
- **Frameworks / Libraries**: Streamlit, asyncio, Pydantic 2, pydantic-ai, httpx, python-dotenv
- **LLMs / APIs**: Google Gemini 2.5 Flash (via pydantic-ai), wttr.in weather API, ip-api/ipapi/wttr geolocation cascade
- **Database / Vector Store**: Not required; in-memory cache handles transient weather data
- **Deployment**: Streamlit runtime (local machine or Streamlit Community Cloud)

---

## Architecture / Approach
- **Experience Layer (`app.py`)**: Streamlit UI with a long-lived asyncio loop, IP-based location detection, multi-day date range pickers, and grouped task rendering. Session state persists user inputs, weather pulls, and previously generated plans for live demos.
- **Reasoning Core (`agent.py`)**: Agentic pipeline (Gemini 2.5 Flash via pydantic-ai) that performs a mandatory feasibility gate, classifies weather relevance, selectively calls weather tools, then orchestrates dual-plan generation (Option A vs. Option B) with explicit risk deltas.
- **Schema & Validation (`models.py`)**: Strongly typed `ChronosResponse`, `PlanOption`, and `TaskStep` models force deterministic control—invalid LLM output is rejected before reaching the UI.
- **Weather Services (`tools.py`)**: Cache-first adapter around wttr.in plus deterministic simulation mode, ensuring continuity even when live forecasts exceed the API window.
- **Intelligence Utilities (`utils.py`)**: Activity classifiers for outdoor sensitivity, rain-aware buffer injectors, IP geolocation cascade, natural-language date parsing, and risk scoring heuristics (Temporal Tetris + Programmatic Empathy).
- **Transparency & Safety**: Decision trace objects reveal why each adjustment happened. A fallback strategy mirrors the presentation deck: if the weather API fails, Chronos auto-switches to simulation without breaking the user flow. (Add an architecture diagram under `assets/` for submission completeness.)

---

## Setup Instructions

### Prerequisites
Before starting, ensure you have the following installed on your laptop:
- **Python 3.10+** ([Download here](https://www.python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/))
- A **Google Gemini API key** ([Get it here](https://aistudio.google.com/app/apikey))

### Quick Start (4 Simple Steps)

#### Step 1: Clone the Repository
Copy the project to your laptop:
```bash
git clone https://github.com/WIBD-Vadodara/HackIconics.git
cd HackIconics
```

#### Step 2: Set Up Python Virtual Environment
Create an isolated environment for the project:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS/Linux (Terminal):**
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

#### Step 4: Configure API Key
Create a `.env` file in the project root and add your Google Gemini API key:
```bash
# Create the .env file
cd HackIconics
# On Windows (PowerShell): echo "GEMINI_API_KEY=your-api-key-here" > .env
# On macOS/Linux: echo "GEMINI_API_KEY=your-api-key-here" > .env
```

Or simply open a text editor, create a file named `.env` in the `HackIconics/` folder and paste:
```
GEMINI_API_KEY=your-api-key-here
```

Replace `your-api-key-here` with your actual Google Gemini API key.

### Run the Application
Once setup is complete, start the application:
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Python command not found | Ensure Python is installed and added to PATH. Restart your terminal. |
| `.venv` not activating | Use `python -m venv .venv` again, then activate it. |
| Module not found error | Ensure your virtual environment is activated and run `pip install -r requirements.txt`. |
| API key errors | Double-check your `.env` file has the correct format: `GEMINI_API_KEY=<your-key>`. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` to use a different port. |

### Optional: Demo Mode
To test the app without an API key (using simulated weather data):
```bash
echo "SIMULATION_MODE=true" >> .env
streamlit run app.py
```

### Optional: Advanced Deployment
For deployment servers, use headless mode:
```bash
streamlit run app.py --server.headless true --server.port 8080
```

---

## Repository Structure
- /app.py → Streamlit UI, session state, and rendering logic
- /agent.py → pydantic-ai agent orchestration and prompt builder
- /models.py → Typed response schema shared between UI and agent
- /tools.py → Weather tooling, caching, and wttr.in adapter
- /utils.py → Location parsing, classification, and risk helpers
- /assets → Branding, screenshots, or architecture diagrams
- /requirements.txt → Python dependencies
- /README.md format.md → Reference template supplied by organizers

---

## Team Members
- Nancy Vaghela – [@nancy325](https://github.com/nancy325) · nancysvaghela@gmail.com
- Bhakti Moteriya – [@Bhakti1112](https://github.com/Bhakti1112) · bhaktimoteriya465@gmail.com
- Isha Patel – [@Isha1530](https://github.com/Isha1530) · ishahp150305@gmail.com
- Arya Mehta – [@aryamehta0302](https://github.com/aryamehta0302) · aryamehta0302@gmail.com

---

## Notes / Assumptions
- Explicit city/state/country inputs always override auto-detect; the agent never invents or infers missing locations.
- wttr.in provides three-day forecasts; for longer horizons, Chronos switches to clearly labeled simulated estimates to avoid misinformation.
- Plans and decision traces live in Streamlit session state only—no external database is provisioned to keep the footprint light for the hackathon.
- IP detection relies on free tiers (ip-api, ipapi, wttr); excessive requests may trigger throttling, so manual entry remains available.
- **Future Vision (from PPT)**: Extend Chronos beyond personal planning into logistics optimization, emergency response scheduling, and field operations coordination wherever weather, time, and critical decisions intersect.
- **Roadmap Ideas**: Calendar integrations, proactive alerts, multi-user persistence, geospatial feasibility datasets, and richer “programmatic empathy” rules for edge-case conditions.

---

## Submission Declaration
This project was developed as part of **WiBD GenAI Hackathon 2026** and all code was written during the hackathon period.
