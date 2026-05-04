import requests
from dotenv import dotenv_values

env = dotenv_values(".env")

# Always get a fresh token using the new email
print("Authenticating with bsccs202362081@mylife.mku.ac.ke...")
r = requests.post("https://acleddata.com/oauth/token", data={
    "username": env.get("ACLED_EMAIL"),
    "password": env.get("ACLED_PASSWORD"),
    "grant_type": "password",
    "client_id": "acled",
    "scope": "authenticated"
})
print(f"Auth status: {r.status_code}")
if r.status_code != 200:
    print(r.text)
    exit()

token = r.json()["access_token"]
print("Got fresh token.")

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

endpoints = {
    "ACLED Events": ("https://acleddata.com/api/acled/read", {"_format": "json", "limit": 2}),
    "CAST Forecasts": ("https://acleddata.com/api/cast/read", {"_format": "json", "limit": 2}),
}

for name, (url, params) in endpoints.items():
    r = requests.get(url, headers=headers, params=params)
    print(f"\n--- {name} ---")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        rows = data.get("data", [])
        if rows:
            print(f"Fields: {list(rows[0].keys())}")
        else:
            print("No data returned")
    else:
        print(r.text[:300])
