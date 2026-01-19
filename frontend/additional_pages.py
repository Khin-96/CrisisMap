def show_statistical_analysis(country: str, date_range):
    st.header("Advanced Statistical Analysis")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Temporal Trends", "Spatial Hotspots", "Actor Networks", "Violence Patterns"]
    )
    
    params = {}
    if country != "All":
        params["country"] = country
    
    if analysis_type == "Temporal Trends":
        params["analysis_type"] = "temporal"
        data = fetch_data("/api/statistical-analysis", params)
        
        if data:
            st.subheader("Temporal Trend Analysis")
            
            # Trend metrics
            trend_data = data["trend_analysis"]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Events Trend Slope", f"{trend_data['events_slope']:.3f}")
            with col2:
                st.metric("Fatalities Trend Slope", f"{trend_data['fatalities_slope']:.3f}")
            with col3:
                st.metric("Events Correlation", f"{trend_data['events_correlation']:.3f}")
            with col4:
                st.metric("Trend Significance", trend_data["events_trend_significance"])
            
            # Temporal data visualization
            temporal_df = pd.DataFrame(data["temporal_data"])
            if not temporal_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temporal_df["period"], 
                    y=temporal_df["events"], 
                    mode='lines+markers', 
                    name='Events'
                ))
                fig.add_trace(go.Scatter(
                    x=temporal_df["period"], 
                    y=temporal_df["fatalities"], 
                    mode='lines+markers', 
                    name='Fatalities',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Temporal Conflict Patterns",
                    xaxis_title="Time Period",
                    yaxis_title="Number of Events",
                    yaxis2=dict(title="Number of Fatalities", overlaying='y', side='right')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Spatial Hotspots":
        params["analysis_type"] = "spatial"
        data = fetch_data("/api/statistical-analysis", params)
        
        if data:
            st.subheader("Spatial Hotspot Analysis")
            
            # Hotspot metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Clusters", data["total_clusters"])
            with col2:
                st.metric("Noise Points", data["noise_points"])
            with col3:
                st.metric("Method Used", data["method"].capitalize())
            
            # Hotspot visualization
            hotspots = data.get("hotspots", [])
            if hotspots:
                # Create map
                if hotspots:
                    avg_lat = sum(h["center_lat"] for h in hotspots) / len(hotspots)
                    avg_lon = sum(h["center_lon"] for h in hotspots) / len(hotspots)
                    
                    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)
                    
                    for hotspot in hotspots:
                        folium.CircleMarker(
                            location=[hotspot["center_lat"], hotspot["center_lon"]],
                            radius=max(5, hotspot["intensity_score"] * 20),
                            popup=f"Cluster {hotspot['cluster_id']}: {hotspot['event_count']} events",
                            color="red",
                            fill=True,
                            fillColor="red"
                        ).add_to(m)
                    
                    st_folium(m, width=700, height=500)
                
                # Hotspot details table
                hotspots_df = pd.DataFrame(hotspots)
                st.dataframe(hotspots_df, use_container_width=True)
    
    elif analysis_type == "Actor Networks":
        params["analysis_type"] = "actors"
        data = fetch_data("/api/statistical-analysis", params)
        
        if data:
            st.subheader("Actor Network Analysis")
            
            # Top actors
            top_actors = data.get("top_actors", [])
            if top_actors:
                actors_df = pd.DataFrame(top_actors)
                
                # Actor activity chart
                fig = px.bar(actors_df, x="actor", y="activity_score", 
                           title="Actor Activity Scores")
                st.plotly_chart(fig, use_container_width=True)
                
                # Actor details
                st.dataframe(actors_df, use_container_width=True)
    
    elif analysis_type == "Violence Patterns":
        params["analysis_type"] = "violence"
        data = fetch_data("/api/statistical-analysis", params)
        
        if data:
            st.subheader("Violence Pattern Analysis")
            
            # Event patterns
            event_patterns = data.get("event_patterns", [])
            if event_patterns:
                patterns_df = pd.DataFrame(event_patterns)
                st.dataframe(patterns_df, use_container_width=True)
            
            # Escalation events
            escalation_events = data.get("escalation_events", [])
            if escalation_events:
                st.subheader("Recent Escalation Events")
                escalation_df = pd.DataFrame(escalation_events)
                st.dataframe(escalation_df, use_container_width=True)
            
            # Intensity distribution
            intensity_dist = data.get("intensity_distribution", {})
            if intensity_dist:
                st.subheader("Violence Intensity Distribution")
                intensity_df = pd.DataFrame(list(intensity_dist.items()), 
                                          columns=["Intensity", "Count"])
                fig = px.pie(intensity_df, values="Count", names="Intensity", 
                           title="Violence Intensity Distribution")
                st.plotly_chart(fig, use_container_width=True)

def show_early_warning(country: str, date_range):
    st.header("Early Warning Indicators")
    
    params = {}
    if country != "All":
        params["country"] = country
    
    # Fetch early warning data
    warning_data = fetch_data("/api/early-warning", params)
    
    if warning_data:
        # Overall risk assessment
        risk_score = warning_data.get("risk_score", 0)
        
        # Risk level display
        if risk_score >= 0.7:
            risk_level = "High"
            risk_color = "[H]"
        elif risk_score >= 0.4:
            risk_level = "Medium"
            risk_color = "[M]"
        else:
            risk_level = "Low"
            risk_color = "[L]"
        
        st.subheader(f"Overall Risk Level: {risk_color} {risk_level}")
        st.progress(risk_score)
        
        # Individual indicators
        st.subheader("Risk Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Activity Spike", f"{warning_data.get('activity_spike', 0):.2f}x")
            st.metric("Geographic Spread", f"{warning_data.get('geographic_spread', 0):.0f} locations")
            st.metric("Actor Diversification", f"{warning_data.get('actor_diversification', 0):.2f}")
        
        with col2:
            st.metric("Fatality Increase", f"{warning_data.get('fatality_increase', 0):.2f}x")
            st.metric("Event Diversification", f"{warning_data.get('event_diversification', 0):.2f}")
            st.metric("Risk Score", f"{risk_score:.2f}")
        
        # Risk interpretation
        st.subheader("Risk Interpretation")
        
        if risk_score >= 0.7:
            st.error("HIGH RISK: Significant escalation indicators detected. Immediate attention recommended.")
        elif risk_score >= 0.4:
            st.warning("MEDIUM RISK: Moderate escalation indicators present. Monitor closely.")
        else:
            st.success("LOW RISK: No significant escalation indicators detected.")
        
        # Recommendations
        st.subheader("Recommended Actions")
        
        if warning_data.get('activity_spike', 0) > 1.5:
            st.info("Increased Monitoring: Recent activity spike detected. Increase monitoring frequency.")
        
        if warning_data.get('fatality_increase', 0) > 1.5:
            st.info("Humanitarian Alert: Significant increase in fatalities. Prepare humanitarian response.")
        
        if warning_data.get('geographic_spread', 0) > 10:
            st.info("Geographic Expansion: Conflict spreading to new areas. Update regional assessments.")
        
        if warning_data.get('actor_diversification', 0) > 0.5:
            st.info("Actor Complexity: New actors involved. Update actor profiles and engagement strategies.")
    
    else:
        st.warning("Unable to fetch early warning data")