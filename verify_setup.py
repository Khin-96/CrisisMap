#!/usr/bin/env python3
"""
Verify CrisisMap setup and dependencies
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    print("🐍 Checking Python packages...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'motor', 'pymongo', 'pandas', 
        'numpy', 'scikit-learn', 'python-multipart'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    return missing

def check_mongodb():
    """Check MongoDB connection"""
    print("\n🍃 Checking MongoDB...")
    
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/Crisis', serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        client.close()
        print("  ✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"  ❌ MongoDB connection failed: {e}")
        return False

def check_nodejs():
    """Check Node.js and npm"""
    print("\n📦 Checking Node.js...")
    
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Node.js {result.stdout.strip()}")
        else:
            print("  ❌ Node.js not found")
            return False
            
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ npm {result.stdout.strip()}")
        else:
            print("  ❌ npm not found")
            return False
            
        return True
    except Exception as e:
        print(f"  ❌ Node.js check failed: {e}")
        return False

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking project files...")
    
    required_files = [
        'backend_v2/main.py',
        'backend_v2/database.py',
        'backend_v2/csv_adapter.py',
        'next-frontend/package.json',
        'requirements_v2.txt',
        '.env'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing.append(file_path)
    
    return missing

def check_csv_files():
    """Check CSV files"""
    print("\n📊 Checking CSV data files...")
    
    csv_dir = Path('csv')
    if csv_dir.exists():
        csv_files = list(csv_dir.glob('*.xlsx')) + list(csv_dir.glob('*.csv'))
        if csv_files:
            for csv_file in csv_files:
                print(f"  ✅ {csv_file}")
            return True
        else:
            print("  ⚠️  No CSV/Excel files found in csv/ directory")
            return False
    else:
        print("  ❌ csv/ directory not found")
        return False

def main():
    """Main verification function"""
    print("🔍 CrisisMap v2.0 Setup Verification")
    print("=" * 50)
    
    # Check Python packages
    missing_packages = check_python_packages()
    
    # Check MongoDB
    mongodb_ok = check_mongodb()
    
    # Check Node.js
    nodejs_ok = check_nodejs()
    
    # Check files
    missing_files = check_files()
    
    # Check CSV files
    csv_ok = check_csv_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_good = True
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements_v2.txt")
        all_good = False
    else:
        print("✅ All Python packages installed")
    
    if not mongodb_ok:
        print("❌ MongoDB not accessible")
        print("   Make sure MongoDB is running on localhost:27017")
        all_good = False
    else:
        print("✅ MongoDB connection working")
    
    if not nodejs_ok:
        print("❌ Node.js/npm not available")
        print("   Install Node.js 18+ from https://nodejs.org/")
        all_good = False
    else:
        print("✅ Node.js/npm available")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        all_good = False
    else:
        print("✅ All required files present")
    
    if not csv_ok:
        print("⚠️  No CSV data files found")
        print("   Add CSV/Excel files to csv/ directory for testing")
    else:
        print("✅ CSV data files available")
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("   You can start CrisisMap with: start_crisismap.bat")
    else:
        print("⚠️  Setup verification FAILED!")
        print("   Please fix the issues above before starting")
    print("=" * 50)

if __name__ == "__main__":
    main()