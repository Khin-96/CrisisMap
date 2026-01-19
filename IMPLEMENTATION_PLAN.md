# CrisisMap Implementation Plan - Phase 1

This plan outlines the steps to upgrade CrisisMap from a prototype/scaffolding to a functional platform meeting all user requirements.

## 1. Data Layer & Ingestion
- [x] **Database Integration**: Add SQLAlchemy and SQLite for local persistence.
- [x] **Live ACLED API**: Implement authentic ACLED API fetching logic.
- [x] **UN Data Ingestion**: Implement displacement data mock/structure.
- [x] **Data Sync Logic**: Logic integrated into the fetch loop.

## 2. Advanced Analytics Wiring
- [x] **Connect ML Predictor**: Wired to API for real forecasting.
- [x] **Connect Anomaly Detector**: Integrated for automated risk spikes.
- [x] **Refine Driver Analysis**: Enabled conflict_driver_analyzer.py for integrated datasets.

## 3. Frontend Wiring (Modern UI)
- [x] **Universal API Client**: Standardized fetch_data helper.
- [x] **Predictions Dashboard**: Implemented forecasting and hotspot migration.
- [x] **Advanced Analysis Tab**: Implemented Actor statistics and Driver analysis.
- [x] **Alert Center**: Functional UI notification system.

## 4. Enhanced Visualizations
- [x] **Actor Network Graph**: Integrated into Analysis page.
- [x] **Correlation Maps**: Hotspot migration and risk maps implemented.

## 5. Reports & Decision Support
- [x] **AI-Powered Policy Briefs**: Functional brief generation UI.
- [x] **Export Center**: Standardized exports via API.

---
**Current Focus**: 1. Data Layer & Ingestion
