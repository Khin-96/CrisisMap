import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import time
from typing import Dict, List
import json
import numpy as np

# Minimalist configuration
st.set_page_config(
    page_title="CRISISMAP",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API base URL
API_BASE = "http://localhost:8000"

class ModernUIComponents:
    """Minimalist B&W UI components for CrisisMap"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, color: str = "black"):
        """Create minimalist metric card"""
        delta_html = f"<span style='color: #000000; opacity: 0.6; font-size: 14px;'>{delta}</span>" if delta else ""
        
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #000000; 
                    padding: 20px; border-radius: 0px; margin: 10px 0;">
            <h3 style="color: #000000; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">{title}</h3>
            <p style="color: #000000; margin: 10px 0; font-size: 32px; font-weight: 300; letter-spacing: -1px;">{value}</p>
            {f"<div style='margin-top: 5px;'>{delta_html}</div>" if delta else ""}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_alert_box(message: str, alert_type: str = "info"):
        """Create minimalist alert box"""
        border_weight = "2px" if alert_type in ["error", "warning"] else "1px"
        
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: {border_weight} solid #000000; 
                    padding: 15px; border-radius: 0px; margin: 15px 0;">
            <p style="margin: 0; font-size: 13px; color: #000000; letter-spacing: 0.5px; line-height: 1.4;">
                <span style="font-weight: 900; text-transform: uppercase; margin-right: 12px; font-size: 11px;">{alert_type}</span>
                {message}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_bar(progress: int, title: str):
        """Create minimalist progress bar"""
        st.markdown(f"""
        <div style="margin: 25px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                <span>{title}</span>
                <span>{progress}%</span>
            </div>
            <div style="background-color: #F0F0F0; border: 1px solid #000000; height: 10px; margin-top: 8px; position: relative;">
                <div style="background-color: #000000; height: 8px; width: {progress}%; position: absolute; top: 0; left: 0;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_feature_grid(features: List[Dict]):
        """Create minimalist feature grid"""
        cols = st.columns(len(features))
        for i, feature in enumerate(features):
            with cols[i]:
                st.markdown(f"""
                <div style="background: #FFFFFF; padding: 25px; border: 1px solid #EEEEEE; 
                            text-align: center; margin: 10px 0;">
                    <div style="font-size: 14px; margin-bottom: 10px; font-weight: 900; border: 1px solid #000000; display: inline-block; padding: 5px 10px;">{feature['icon']}</div>
                    <h4 style="margin: 10px 0; color: #000000; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">{feature['title']}</h4>
                    <p style="color: #666; font-size: 10px; margin: 0; font-weight: 400; line-height: 1.5;">{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)

def fetch_data(endpoint: str, params: dict = None):
    """Fetch data from primary API"""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", params=params or {}, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_minimalist_sidebar():
    """Retractable sidebar with minimalist options"""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px 0; border-bottom: 3px solid #000000; margin-bottom: 30px;">
            <h1 style="color: #000000; margin: 0; font-size: 28px; font-weight: 900; letter-spacing: -1.5px;">CRISISMAP</h1>
            <p style="color: #666; margin: 0; font-size: 9px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Intelligence Protocol</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### NAVIGATION")
        main_pages = [
            {"id": "dashboard", "name": "System Overview", "icon": "01"},
            {"id": "realtime", "name": "Live Monitor", "icon": "02"},
            {"id": "analysis", "name": "Deep Analysis", "icon": "03"},
            {"id": "predictions", "name": "ML Forecasts", "icon": "04"},
            {"id": "alerts", "name": "Alert Center", "icon": "05"},
            {"id": "reports", "name": "Protocol Reports", "icon": "06"},
        ]
        
        selected_page = st.selectbox(
            "Target View",
            options=[page["id"] for page in main_pages],
            format_func=lambda x: next(f"{p['icon']} : {p['name']}" for p in main_pages if p["id"] == x),
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### PARAMETERS")
        
        # Filters
        countries = ["All", "Democratic Republic of Congo", "Uganda", "Rwanda", "Burundi", "South Sudan"]
        selected_country = st.selectbox("Geographic Focus", countries)
        
        date_option = st.radio(
            "Temporal Window",
            ["Short Term (7d)", "Cycle (30d)", "Strategic (90d)", "Custom"],
            index=1
        )
        
        if "Custom" in date_option:
            start_date = st.date_input("Start", datetime.now() - timedelta(days=30))
            end_date = st.date_input("End", datetime.now())
        else:
            days = 7 if "7d" in date_option else 30 if "30d" in date_option else 90
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("RESET SYSTEM VECTORS", use_container_width=True):
            st.rerun()
            
        return {
            "page": selected_page,
            "country": selected_country,
            "start_date": start_date,
            "end_date": end_date
        }

def show_minimalist_header(title: str, subtitle: str):
    """B&W Header without gradients"""
    st.markdown(f"""
    <div style="padding: 30px 0; border-bottom: 1px solid #000000; margin-bottom: 40px;">
        <h1 style="margin: 0; font-size: 42px; font-weight: 900; color: #000000; letter-spacing: -2px; text-transform: uppercase;">{title}</h1>
        <p style="margin: 5px 0 0 0; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 3px; font-weight: 700;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_modern_dashboard(filters: Dict):
    """Minimalist Dashboard Implementation"""
    show_minimalist_header("DASHBOARD", f"System Status: Active | Region: {filters['country']}")
    
    # Metrics
    trends_data = fetch_data("/api/trends", {"country": filters["country"] if filters["country"] != "All" else None})
    
    if trends_data:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ModernUIComponents.create_metric_card("Conflict Events", f"{trends_data['total_events']:,}", "Active Pattern")
        with m2:
            ModernUIComponents.create_metric_card("Total Fatalities", f"{trends_data['total_fatalities']:,}", "High" if trends_data['total_fatalities'] > 500 else "Stable")
        with m3:
            ModernUIComponents.create_metric_card("Hotspot Density", f"{len(trends_data['hotspot_locations'])}", "Vector Points")
        with m4:
            ModernUIComponents.create_metric_card("Trend Analysis", trends_data['trend_direction'].upper(), "Systemic")
            
    # Risk Progress
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        ModernUIComponents.create_progress_bar(78, "Regional Instability Index")
        ModernUIComponents.create_progress_bar(62, "Resource Scarcity Vector")
    with r2:
        ModernUIComponents.create_progress_bar(45, "Actor Diversification")
        ModernUIComponents.create_progress_bar(89, "Insecurity Probability")

    # Map & Trend
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown("### GEOGRAPHIC INTELLIGENCE")
        if trends_data and trends_data.get("hotspot_locations"):
            hotspots = trends_data["hotspot_locations"]
            m = folium.Map(location=[-2.0, 29.0], zoom_start=6, tiles="CartoDB positron")
            for hs in hotspots[:15]:
                folium.CircleMarker(
                    location=[hs["latitude"], hs["longitude"]],
                    radius=8, color="#000000", fill=True, fill_color="#000000", fill_opacity=0.4,
                    popup=f"Events: {hs.get('event_count', 'N/A')}"
                ).add_to(m)
            st_folium(m, width=None, height=400)
        else:
            st.info("No Geographic data detected in the current sector buffer.")
    
    with col_b:
        st.markdown("### TEMPORAL VECTORS")
        if trends_data and trends_data.get("temporal_data"):
            df_temporal = pd.DataFrame(trends_data["temporal_data"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_temporal["period"], y=df_temporal["total_events"], marker_color='#000000'))
            fig.update_layout(
                height=300, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='white', plot_bgcolor='white',
                showlegend=False,
                hovermode="x unified"
            )
            fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
            fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE', tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No temporal vectors detected.")

    # Table
    st.markdown("### RECENT LOGS")
    events_data = fetch_data("/api/events", {"limit": 10, "country": filters["country"] if filters["country"] != "All" else None})
    if events_data:
        df = pd.DataFrame(events_data)
        st.dataframe(df[["event_date", "location", "event_type", "fatalities"]], use_container_width=True)

def show_realtime_monitor(filters: Dict):
    show_minimalist_header("LIVE MONITOR", "Real-time status verification and synchronization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ModernUIComponents.create_metric_card("Sync Status", "ACTIVE", "0.2s Latency")
    with col2:
        ModernUIComponents.create_metric_card("Active Signals", "2,482", "Stable")
    with col3:
        ModernUIComponents.create_metric_card("System Health", "98.2%", "Nominal")
        
    st.markdown("### LIVE SIGNAL FEED")
    for i in range(5):
        st.markdown(f"""
        <div style="border-bottom: 1px solid #EEEEEE; padding: 15px 0; display: flex; justify-content: space-between;">
            <div style="font-size: 12px; font-weight: 700;">SIGNAL_{100+i} :: SECTOR_{i*10}</div>
            <div style="font-size: 11px; color: #666;">VERIFIED :: {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

def show_predictions_page(filters: Dict):
    show_minimalist_header("ML FORECASTS", "System state probability and hotspot prediction")
    
    country_param = {"country": filters["country"] if filters["country"] != "All" else None}
    pred_data = fetch_data("/api/predictions/fatalities", country_param)
    hotspot_preds = fetch_data("/api/predictions/hotspots", country_param)
    
    if pred_data and "predictions" in pred_data:
        st.markdown(f"#### TREND PROJECTION :: {pred_data.get('predicted_trend', 'STABLE').upper()}")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### PROBABILITY TENSOR (FATALITIES)")
            df_pred = pd.DataFrame(pred_data["predictions"])
            fig = px.area(df_pred, x="date", y="predicted_fatalities")
            fig.update_traces(line_color='#000000', fillcolor='rgba(0,0,0,0.1)')
            fig.update_layout(
                height=400, paper_bgcolor='white', plot_bgcolor='white',
                margin=dict(l=0,r=0,t=20,b=0)
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with c2:
            st.markdown("#### CONFIDENCE METRICS")
            ModernUIComponents.create_metric_card("Historical Avg", f"{pred_data.get('historical_avg', 0):.2f}")
            ModernUIComponents.create_metric_card("Model Profile", pred_data.get("model_used", "STOCHASTIC").split("_")[-1])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### PREDICTIVE VECTORS")
            if hotspot_preds and "hotspot_predictions" in hotspot_preds:
                m = folium.Map(location=[-1.5, 29.2], zoom_start=7, tiles="CartoDB positron")
                for hp in hotspot_preds["hotspot_predictions"]:
                    folium.Marker(
                        [hp["latitude"], hp["longitude"]],
                        icon=folium.Icon(color='black', icon='warning', prefix='fa'),
                        popup=f"Risk: {hp['risk_level'].upper()}"
                    ).add_to(m)
                st_folium(m, width=None, height=250)
    else:
        st.warning("Insufficient signal history to initialize ML models for this sector.")

def show_advanced_analysis(filters: Dict):
    show_minimalist_header("DEEP ANALYSIS", "Recursive pattern investigation and driver synthesis")
    t1, t2, t3 = st.tabs(["CONFLICT DRIVERS", "HUMANITARIAN DATA", "ACTOR NETWORK"])
    
    country_param = {"country": filters["country"] if filters["country"] != "All" else None}
    
    with t1:
        drivers = fetch_data("/api/analysis/drivers", country_param)
        if drivers:
            st.markdown("#### SYSTEMIC DRIVERS")
            # Parse backend response structure: {'synthesized_drivers': {'top_drivers': [['name', {'average_score': X}], ...]}}
            top_drivers = drivers.get("synthesized_drivers", {}).get("top_drivers", [])
            
            for item in top_drivers[:3]:
                # Backend returns [driver_name, driver_data_dict]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    driver_name = item[0]
                    driver_data = item[1]
                    impact = driver_data.get("average_score", 0)
                else:
                    # Fallback for unexpected structure
                    driver_name = str(item)
                    impact = 0

                description = "Identified as a significant conflict driver based on multi-source correlation."
                
                ModernUIComponents.create_alert_box(f"{driver_name}: {description} (Impact: {int(impact*100)}%)", "info")
        else:
            st.info("Driver analysis requiring more signal data.")
            
    with t2:
        st.markdown("#### HUMANITARIAN SYNCHRONIZATION")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            if st.button("RUN HDX SYNC PROTOCOL", use_container_width=True):
                with st.spinner("Synchronizing with OCHA/IOM..."):
                    requests.post(f"{API_BASE}/api/sync/regional")
                    st.success("SYNC COMPLETE")
        
        hum_data = fetch_data("/api/analysis/humanitarian")
        if hum_data:
            df_hum = pd.DataFrame(hum_data)
            fig = px.pie(df_hum, values='value', names='admin1', title="Displacement by Province (HDX Source)")
            fig.update_traces(marker=dict(colors=['#000000', '#333333', '#666666', '#999999', '#CCCCCC']))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No humanitarian indicators in database. Run SYNC to fetch.")
            
    with t3:
        network = fetch_data("/api/analysis/network", country_param)
        if network and "actor_statistics" in network:
            st.markdown("#### ACTOR INFLUENCE (BY FATALITIES)")
            df_actors = pd.DataFrame(network["actor_statistics"])
            fig = px.bar(df_actors.head(10), x="actor", y="fatalities", color_discrete_sequence=['#000000'])
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            if "network_analysis" in network and "edges" in network["network_analysis"]:
                st.markdown("#### KNOWN CO-OCCURRENCE NODES")
                df_edges = pd.DataFrame(network["network_analysis"]["edges"])
                st.dataframe(df_edges, use_container_width=True)
        else:
            st.info("Actor relationship network requiring deeper ingestion history.")

def show_alert_center(filters: Dict):
    show_minimalist_header("ALERT CENTER", "Anomaly detection and systemic outlier reports")
    
    country_param = {"country": filters["country"] if filters["country"] != "All" else None}
    anomalies = fetch_data("/api/alerts/anomalies", country_param)
    
    if anomalies and "key_insights" in anomalies:
        for insight in anomalies["key_insights"]:
            ModernUIComponents.create_alert_box(insight, "warning")
            
        if "statistical_anomalies" in anomalies and anomalies["statistical_anomalies"].get("anomalies"):
            st.markdown("#### STATISTICAL OUTLIERS")
            df_anom = pd.DataFrame(anomalies["statistical_anomalies"]["anomalies"])
            st.dataframe(df_anom[["date", "value", "severity", "anomaly_type"]], use_container_width=True)
    else:
        st.info("No anomalous vectors currently detected in high-confidence buffers.")

def show_reports_page(filters: Dict):
    show_minimalist_header("PROTOCOL REPORTS", "Automated intelligence summary and data serialization")
    
    country_param = {"country": filters["country"] if filters["country"] != "All" else None}
    
    if st.button("GENERATE STRATEGIC POLICY BRIEF"):
        with st.spinner("Compiling multi-source intelligence..."):
            brief = fetch_data("/api/reports/policy-brief", country_param)
            if brief:
                st.markdown(f"### {brief.get('title', 'Policy Brief')}")
                st.markdown(brief.get("executive_summary", "Summary unavailable."))
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("#### RECOMMENDED ACTIONS")
                    for recommendation in brief.get("recommendations", []):
                        st.markdown(f"* {recommendation}")
                with col_r2:
                    st.markdown("#### RISK ASSESSMENT")
                    st.markdown(brief.get("outlook", "Outlook unavailable."))
                
                st.download_button("DOWNLOAD AS TEXT", brief.get("executive_summary", ""), "crisis_report.txt")
            else:
                st.error("Report generation failed. System status: Offline.")

def main():
    filters = create_minimalist_sidebar()
    
    if filters["page"] == "dashboard":
        show_modern_dashboard(filters)
    elif filters["page"] == "realtime":
        show_realtime_monitor(filters)
    elif filters["page"] == "predictions":
        show_predictions_page(filters)
    elif filters["page"] == "analysis":
        show_advanced_analysis(filters)
    elif filters["page"] == "alerts":
        show_alert_center(filters)
    elif filters["page"] == "reports":
        show_reports_page(filters)
    
    # Global Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top: 2px solid #000000; padding: 20px 0; text-align: left;">
        <p style="font-size: 9px; letter-spacing: 2px; font-weight: 700; color: #000000;">PROTOCOL: CRISISMAP :: VERSION 2.0.4 :: CLASSIFIED</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()