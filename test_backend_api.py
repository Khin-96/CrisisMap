#!/usr/bin/env python3
"""
Test script to verify backend API endpoints are working
"""

import requests
import json

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/api/",
        "/api/events?limit=5",
        "/api/trends",
        "/api/hotspots",
        "/api/system/status"
    ]
    
    print("🔍 Testing Backend API Endpoints")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            print(f"\n📡 Testing: {endpoint}")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                
                if endpoint == "/api/events?limit=5":
                    print(f"   📊 Events returned: {len(data)}")
                    if data:
                        print(f"   📍 Sample event: {data[0].get('event_type', 'N/A')} in {data[0].get('location', 'N/A')}")
                
                elif endpoint == "/api/trends":
                    print(f"   📈 Total events: {data.get('total_events', 0)}")
                    print(f"   💀 Total fatalities: {data.get('total_fatalities', 0)}")
                    print(f"   🔥 Hotspots: {len(data.get('hotspot_locations', []))}")
                
                elif endpoint == "/api/hotspots":
                    hotspots = data.get('hotspots', [])
                    print(f"   🎯 Hotspots found: {len(hotspots)}")
                
                elif endpoint == "/api/system/status":
                    print(f"   🖥️  Backend: {data.get('backend', 'unknown')}")
                    print(f"   🗄️  Database: {data.get('database', 'unknown')}")
                    print(f"   📊 Total events: {data.get('dataStats', {}).get('totalEvents', 0)}")
                
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"   Error: {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed - Is the backend running?")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 API Testing Complete")

if __name__ == "__main__":
    test_api_endpoints()