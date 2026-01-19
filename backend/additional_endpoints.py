@app.get("/api/statistical-analysis")
async def get_statistical_analysis(
    country: Optional[str] = None,
    analysis_type: str = "temporal"
):
    """Get advanced statistical analysis"""
    try:
        # Fetch and process data
        df = data_ingestion.fetch_acled_data(country=country or "DR Congo")
        df = data_ingestion.validate_data(df)
        statistical_analyzer.load_data(df)
        
        # Perform analysis based on type
        if analysis_type == "temporal":
            result = statistical_analyzer.temporal_trend_analysis()
        elif analysis_type == "spatial":
            result = statistical_analyzer.spatial_hotspot_analysis()
        elif analysis_type == "actors":
            result = statistical_analyzer.actor_network_analysis()
        elif analysis_type == "violence":
            result = statistical_analyzer.violence_pattern_analysis()
        elif analysis_type == "early_warning":
            result = statistical_analyzer.early_warning_indicators()
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/early-warning")
async def get_early_warning(
    country: Optional[str] = None
):
    """Get early warning indicators for conflict escalation"""
    try:
        # Fetch and process data
        df = data_ingestion.fetch_acled_data(country=country or "DR Congo")
        df = data_ingestion.validate_data(df)
        statistical_analyzer.load_data(df)
        
        # Calculate early warning indicators
        indicators = statistical_analyzer.early_warning_indicators()
        
        return indicators
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)