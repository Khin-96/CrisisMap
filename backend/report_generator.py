import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import io
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

class ReportGenerator:
    """Generate exportable analytical reports"""
    
    def __init__(self):
        self.data = None
    
    def load_data(self, df: pd.DataFrame):
        """Load data for report generation"""
        self.data = df.copy()
    
    def generate_summary_report(self, country: str = "DR Congo", period: str = "monthly") -> Dict:
        """Generate comprehensive summary report"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Basic statistics
        total_events = len(self.data)
        total_fatalities = int(self.data["fatalities"].sum())
        date_range = f"{self.data['event_date'].min().strftime('%Y-%m-%d')} to {self.data['event_date'].max().strftime('%Y-%m-%d')}"
        
        # Top locations
        top_locations = self.data.groupby("location").size().nlargest(5).to_dict()
        
        # Top actors
        top_actors = self.data.groupby("actor1").size().nlargest(5).to_dict()
        
        # Event type distribution
        event_types = self.data["event_type"].value_counts().to_dict()
        
        # Monthly trends
        self.data["month"] = self.data["event_date"].dt.to_period("M")
        monthly_trends = self.data.groupby("month").agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        
        return {
            "report_metadata": {
                "country": country,
                "period": period,
                "date_range": date_range,
                "generated_at": datetime.now().isoformat(),
                "total_events": total_events,
                "total_fatalities": total_fatalities
            },
            "key_findings": {
                "top_locations": top_locations,
                "top_actors": top_actors,
                "event_type_distribution": event_types,
                "monthly_trends": monthly_trends.to_dict("records")
            }
        }
    
    def export_to_csv(self, data: pd.DataFrame, filename: str = None) -> StreamingResponse:
        """Export data to CSV format"""
        if filename is None:
            filename = f"crisismap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create CSV in memory
        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    def export_to_json(self, data: Dict, filename: str = None) -> StreamingResponse:
        """Export data to JSON format"""
        if filename is None:
            filename = f"crisismap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON
        json_data = json.dumps(data, indent=2, default=str)
        
        return StreamingResponse(
            io.BytesIO(json_data.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    def export_to_excel(self, data: Dict, filename: str = None) -> StreamingResponse:
        """Export data to Excel format"""
        if filename is None:
            filename = f"crisismap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create Excel workbook
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            metadata = data["report_metadata"]
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Key findings sheets
            if "top_locations" in data["key_findings"]:
                locations_df = pd.DataFrame(list(data["key_findings"]["top_locations"].items()), 
                                           columns=["Location", "Events"])
                locations_df.to_excel(writer, sheet_name="Top Locations", index=False)
            
            if "top_actors" in data["key_findings"]:
                actors_df = pd.DataFrame(list(data["key_findings"]["top_actors"].items()), 
                                       columns=["Actor", "Events"])
                actors_df.to_excel(writer, sheet_name="Top Actors", index=False)
            
            if "event_type_distribution" in data["key_findings"]:
                event_types_df = pd.DataFrame(list(data["key_findings"]["event_type_distribution"].items()), 
                                            columns=["Event Type", "Count"])
                event_types_df.to_excel(writer, sheet_name="Event Types", index=False)
            
            if "monthly_trends" in data["key_findings"]:
                trends_df = pd.DataFrame(data["key_findings"]["monthly_trends"])
                trends_df.to_excel(writer, sheet_name="Monthly Trends", index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    def generate_policy_brief(self, country: str = "DR Congo") -> Dict:
        """Generate policy brief format report"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Recent trends (last 30 days)
        recent_cutoff = datetime.now() - pd.Timedelta(days=30)
        recent_data = self.data[self.data["event_date"] >= recent_cutoff]
        
        # Key metrics
        recent_events = len(recent_data)
        recent_fatalities = int(recent_data["fatalities"].sum())
        
        # Emerging hotspots
        emerging_hotspots = recent_data.groupby("location").size().nlargest(3).to_dict()
        
        # Risk assessment
        if recent_events > 100:
            risk_level = "High"
        elif recent_events > 50:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "policy_brief": {
                "title": f"Conflict Analysis Brief: {country}",
                "date": datetime.now().strftime("%B %d, %Y"),
                "executive_summary": f"In the past 30 days, {country} has experienced {recent_events} conflict events resulting in {recent_fatalities} fatalities. The current risk level is assessed as {risk_level}.",
                "key_developments": [
                    f"Total conflict events: {recent_events}",
                    f"Total fatalities: {recent_fatalities}",
                    f"Emerging hotspots: {', '.join(emerging_hotspots.keys())}",
                    f"Risk level: {risk_level}"
                ],
                "recommendations": [
                    "Increase monitoring in identified hotspot areas",
                    "Engage with key actors in conflict zones",
                    "Prepare humanitarian response for high-risk areas",
                    "Support community-based early warning systems"
                ],
                "data_sources": ["ACLED", "UN datasets", "Local monitoring"],
                "contact": "CrisisMap Analysis Team"
            }
        }