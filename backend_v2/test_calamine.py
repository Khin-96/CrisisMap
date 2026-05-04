import pandas as pd
import os

file_path = r'c:\Users\ADMIN\Desktop\Projects\CrisisMap\crisismap\CAST_PREDS\cast_static_2026-04-24.xlsx'

try:
    print(f"Testing pandas {pd.__version__} with calamine engine...")
    df = pd.read_excel(file_path, engine='calamine')
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
