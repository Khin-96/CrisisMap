import requests
import os
import time
import sys

# Ensure output is not buffered
sys.stdout.reconfigure(line_buffering=True)

file_path = r'c:\Users\ADMIN\Desktop\Projects\CrisisMap\crisismap\CAST_PREDS\cast_static_2026-04-24.xlsx'
url = 'http://127.0.0.1:8000/api/upload/cast-csv'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

print(f"Uploading {file_path} to {url}...")

try:
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        response = requests.post(url, files=files, timeout=30)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        fetch_id = data.get('fetch_id')
        print(f"Upload successful! Fetch ID: {fetch_id}")
        
        status_url = f"http://127.0.0.1:8000/api/cast/fetch/{fetch_id}"
        print(f"Polling status at {status_url}...")
        
        for i in range(60): 
            status_resp = requests.get(status_url, timeout=10)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                status = status_data.get('status')
                message = status_data.get('message')
                print(f"[{i}] Status: {status} - {message}")
                
                if status == 'completed':
                    print("✅ Processing finished successfully!")
                    break
                elif status == 'error':
                    print(f"❌ Processing failed: {message}")
                    break
            else:
                print(f"Failed to get status: {status_resp.status_code}")
                break
            time.sleep(5)
    else:
        print(f"Upload failed: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error occurred: {e}")
