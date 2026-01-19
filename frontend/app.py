import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import folium
from streamlit_folium import st_folium
from typing import Optional, List, Dict

# UI helpers (imported patch)
from ui_helpers import metric_card_html, story_panel_html, render_hotspots_map

# API base
API_BASE = "http://localhost:8000"


def fetch_data(endpoint: str, params: Optional[dict] = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params or {}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

# Theming
LIGHT_THEME = {"bg": "#ffffff", "fg": "#111111", "card": "#f6f7fb", "primary": "#2563eb"}
DARK_THEME = {"bg": "#0b1020", "fg": "#e5e7eb", "card": "#141a2a", "primary": "#4EC9B0"}

def inject_theme(dark: bool):
    t = DARK_THEME if dark else LIGHT_THEME
    css = f"""
    <style>
      :root {{ --bg: {t['bg']}; --fg: {t['fg']}; --card: {t['card']}; --primary: {t['primary']}; }}
      body {{ background: var(--bg); color: var(--fg); font-family: Arial, sans-serif; }}
      .header {{ background: linear-gradient(90deg, var(--primary), #6ee7b7); padding: 16px; border-radius: 12px; color: white; text-align: center; margin-bottom: 12px; }}
      .card {{ background: var(--card); border-radius: 12px; padding: 14px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); color: var(--fg); }}
      .card-title { font-size: 12px; opacity: 0.9; margin-bottom: 6px; }
      .card-value { font-size: 22px; font-weight: 700; }
      .story { background: var(--card); padding: 12px; border-radius: 8px; margin-top: 8px; }
      .section-title { font-family: Arial, sans-serif; font-size: 14px; font-weight: 700; margin: 8px 0; }
      .footer { text-align: center; padding: 12px; color: var(--fg); font-size: 12px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

STORY_STEPS = [
    {"title": "Overview", "text": "A high-level view of conflict activity, hotspots, and trends.", "image": None},
    {"title": "Trends", "text": "Explore monthly trends for events and fatalities.", "image": None},
    {"title": "Hotspots", "text": "Geographic concentration of violence and intensity.", "image": None},
    {"title": "Actions", "text": "Export current view or share insights with stakeholders.", "image": None},
]

# Theme helpers

def header_banner():
    st.markdown("""
    <div class='header' aria-label='CrisisMap header'>CrisisMap - Conflict Early Warning & Trend Analysis</div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="CrisisMap", layout="wide")
    if 'dark' not in st.session_state:
        st.session_state.dark = False
    dark = st.sidebar.checkbox("Dark theme", value=st.session_state.dark, key='dark_theme')
    st.session_state.dark = dark
    inject_theme(dark)

    header_banner()

    # Left filters and story in a two-column layout
    with st.container():
        left, right = st.columns([1, 2], gap='large')
        with left:
            st.markdown("### Filters & Data")
            country = st.selectbox("Country", ["All", "Democratic Republic of Congo", "Uganda", "Rwanda", "Burundi"], key="country")
            start = st.date_input("Start date", value=date.today() - timedelta(days=365), key='start')
            end = st.date_input("End date", value=date.today(), key='end')
            sources = st.multiselect("Data sources", ["ACLED", "UN datasets"], default=["ACLED"], key="sources")
            event_types = st.multiselect("Event types", ["Battle", "Violence against civilians", "Remote violence", "Protests"], default=["Battle"], key="types")
            st.markdown("---")
            st.markdown("## Story")
            if 'story_idx' not in st.session_state:
                st.session_state.story_idx = 0
            idx = st.session_state.story_idx
            st.markdown(story_panel_html(STORY_STEPS, idx), unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="story_back_main"):
                    st.session_state.story_idx = max(0, idx - 1)
            with col2:
                if st.button("Next", key="story_next_main"):
                    st.session_state.story_idx = min(len(STORY_STEPS) - 1, idx + 1)
        with right:
            st.markdown("### Overview")
            trends = fetch_data("/api/trends", {"country": country if country != 'All' else None}) if country else fetch_data("/api/trends")
            total_events = trends.get('total_events', 0) if trends else 0
            total_fatalities = trends.get('total_fatalities', 0) if trends else 0
            hotspot_locations = trends.get('hotspot_locations', []) if trends else []
            trend_dir = trends.get('trend_direction', 'stable') if trends else 'stable'

            # 4 metric cards using HTML helper
            st.markdown(metric_card_html("Total Events", f"{total_events:,}"), unsafe_allow_html=True)
            st.markdown(metric_card_html("Total Fatalities", f"{total_fatalities:,}"), unsafe_allow_html=True)
            st.markdown(metric_card_html("Active Hotspots", f"{len(hotspot_locations)}"), unsafe_allow_html=True)
            st.markdown(metric_card_html("Trend Direction", trend_dir.capitalize()), unsafe_allow_html=True)

            st.markdown("### Trends")
            if trends and trends.get('temporal_data'):
                df = pd.DataFrame(trends['temporal_data'])
                if 'period' in df.columns:
                    df['period'] = pd.to_datetime(df['period'])
                    fig = go.Figure()
                    if 'total_events' in df.columns:
                        fig.add_trace(go.Scatter(x=df['period'], y=df['total_events'], mode='lines+markers', name='Events'))
                    if 'total_fatalities' in df.columns:
                        fig.add_trace(go.Scatter(x=df['period'], y=df['total_fatalities'], mode='lines+markers', name='Fatalities', yaxis='y2'))
                        fig.update_layout(yaxis2=dict(title='Fatalities', overlaying='y', side='right'))
                    fig.update_layout(title='Temporal Trends', xaxis_title='Date', yaxis_title='Events')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Demo plot if API data missing
                demo_months = pd.date_range(end=datetime.now(), periods=12, freq='M')
                demo_events = [10 * (i+1) for i in range(12)]
                demo_fatalities = [100 + i*25 for i in range(12)]
                demo_fig = go.Figure()
                demo_fig.add_trace(go.Scatter(x=demo_months, y=demo_events, mode='lines+markers', name='Events'))
                demo_fig.add_trace(go.Scatter(x=demo_months, y=demo_fatalities, mode='lines+markers', name='Fatalities', yaxis='y2'))
                demo_fig.update_layout(title='Temporal Trends (Demo)', xaxis_title='Date', yaxis_title='Events', yaxis2=dict(title='Fatalities', overlaying='y', side='right'))
                st.plotly_chart(demo_fig, use_container_width=True)

            st.markdown("### Hotspots")
            hotspots = trends.get('hotspot_locations', []) if trends else []
            hotspot_map = render_hotspots_map(hotspots[:10])
            if hotspot_map:
                st_folium(hotspot_map, width=700, height=400)
            else:
                st.info("No hotspot data available")

            st.markdown("### Recent Events")
            events = fetch_data("/api/events", {"country": country if country != 'All' else None, "start_date": start, "end_date": end, "limit": 10}) or []
            if isinstance(events, list) and len(events) > 0:
                df_e = pd.DataFrame(events)
                cols = ["event_date","location","event_type","actor1","fatalities"]
                avail = [c for c in cols if c in df_e.columns]
                st.dataframe(df_e[avail], height=260)
            else:
                st.info("No recent events to display.")

    st.markdown("---")
    st.markdown("<div class='footer'>CrisisMap Frontend - UI scaffolding</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
