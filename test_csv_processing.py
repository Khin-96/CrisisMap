#!/usr/bin/env python3
"""
Test script to verify CSV processing with actual data files
"""

import sys
import os
sys.path.append('backend_v2')

from csv_adapter import CSVAdapter
import pandas as pd

def test_csv_files():
    """Test CSV processing with the actual files"""
    csv_adapter = CSVAdapter()
    
    # Test files in csv directory
    csv_files = [
        'csv/Africa_aggregated_data_up_to-2026-02-28.xlsx',
        'csv/number_of_political_violence_events_by_country-year_as-of-27Feb2026.xlsx',
        'csv/number_of_reported_fatalities_by_country-year_as-of-27Feb2026.xlsx'
    ]
    
    for file_path in csv_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Testing: {file_path}")
            print('='*60)
            
            try:
                # Analyze the file
                analysis = csv_adapter.analyze_csv(file_path)
                
                print(f"Total rows: {analysis['total_rows']:,}")
                print(f"Columns: {len(analysis['columns'])}")
                print(f"Detected mappings: {analysis['detected_mappings']}")
                print(f"Missing required: {analysis['missing_required']}")
                
                # Show column suggestions
                suggestions = csv_adapter.get_column_suggestions(analysis['columns'])
                print(f"\nColumn suggestions:")
                for standard_col, matches in suggestions.items():
                    if matches:
                        print(f"  {standard_col}: {matches}")
                
                # Show sample data
                print(f"\nSample columns: {analysis['columns'][:10]}")
                
                # If we can process it automatically, try it
                if len(analysis['missing_required']) == 0:
                    print(f"\n✅ Can process automatically!")
                    
                    # Process a small sample
                    df_processed = csv_adapter.process_csv(file_path)
                    print(f"Processed {len(df_processed)} records")
                    print(f"Columns after processing: {list(df_processed.columns)}")
                    
                    # Show validation
                    validation = csv_adapter.validate_processed_data(df_processed)
                    print(f"Validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
                    if validation['errors']:
                        print(f"Errors: {validation['errors']}")
                    
                else:
                    print(f"\n⚠️  Manual mapping required for: {analysis['missing_required']}")
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    test_csv_files()