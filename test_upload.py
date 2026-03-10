#!/usr/bin/env python3
"""
Test script to upload CSV data directly to the backend
"""

import sys
import os
sys.path.append('backend_v2')

import asyncio
from data_processor import DataProcessor

async def test_upload():
    """Test uploading the Africa dataset"""
    try:
        processor = DataProcessor()
        
        # Test with the Africa dataset
        csv_file = 'csv/Africa_aggregated_data_up_to-2026-02-28.xlsx'
        
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            return
        
        print(f"📊 Testing upload of: {csv_file}")
        
        # Analyze the file first
        print("🔍 Analyzing file structure...")
        analysis = await processor.analyze_csv_file(csv_file)
        
        print(f"✅ Analysis complete:")
        print(f"   - Total rows: {analysis['total_rows']:,}")
        print(f"   - Columns: {len(analysis['columns'])}")
        print(f"   - Missing required: {analysis['missing_required']}")
        
        if len(analysis['missing_required']) == 0:
            print("🚀 Processing file...")
            
            # Process the file
            records_processed = await processor.process_csv_data(
                csv_file, 
                custom_mappings=None, 
                upload_id="test_upload_001"
            )
            
            print(f"✅ Successfully processed {records_processed:,} records!")
            
        else:
            print(f"⚠️  Manual mapping required for: {analysis['missing_required']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_upload())