# Migration Guide: Updated main.py with Full Pipeline

## What Changed

The new `main_new.py` integrates the complete data processing pipeline:

### Old Flow
```
Upload CSV → Basic validation → Store in DB
```

### New Flow
```
Upload CSV → Read file → Validate → Enrich → Analyze → Store → Generate Report
```

## New Features

1. **Full Data Pipeline**
   - File reading and validation
   - Data enrichment with derived features
   - Comprehensive analysis (trends, hotspots, anomalies)
   - Report generation and storage

2. **Analysis Report**
   - Temporal trends
   - Geographic hotspots
   - Anomaly detection
   - Actor analysis
   - Event type distribution

3. **Real-time Updates**
   - WebSocket notifications on completion
   - Progress tracking (0-100%)
   - Detailed status messages

4. **Better Error Handling**
   - Graceful failure with error messages
   - Automatic cleanup
   - Detailed logging

## Migration Steps

### 1. Backup Current main.py
```bash
cp backend_v2/main.py backend_v2/main_backup.py
```

### 2. Replace with New Version
```bash
cp backend_v2/main_new.py backend_v2/main.py
```

### 3. Verify Dependencies
Ensure these files exist:
- `database.py` (MongoDB)
- `ml_pipeline.py`
- `data_processor_v2.py`
- `websocket_manager.py`
- `file_processor.py`
- `analysis_engine.py`

### 4. Test Upload
```bash
# Start backend
python -m uvicorn backend_v2.main:app --reload

# Upload test file
curl -X POST http://localhost:8000/api/upload/csv \
  -F "file=@test_data.csv"

# Check status
curl http://localhost:8000/api/upload/status/{upload_id}
```

## API Changes

### Upload Endpoint
**Request:** Same as before
```bash
POST /api/upload/csv
Content-Type: multipart/form-data
file: <CSV file>
```

**Response:** Same format
```json
{
  "upload_id": "uuid",
  "status": "processing",
  "message": "File uploaded successfully..."
}
```

### Status Endpoint
**New Response Format:**
```json
{
  "status": "completed",
  "progress": 100,
  "message": "Successfully processed 1247 records",
  "records_processed": 1247,
  "analysis": {
    "hotspots": 12,
    "anomalies": 8,
    "trends": 12,
    "actors": 45
  }
}
```

### New Analysis Endpoints
```bash
# Get analysis report
GET /api/upload/status/{upload_id}

# Get trends
GET /api/trends?country=DRC&period=monthly

# Get hotspots
GET /api/hotspots?country=DRC

# Get anomalies
GET /api/analytics/anomalies?country=DRC&days_back=30
```

## WebSocket Events

### Upload Completed
```json
{
  "type": "upload_completed",
  "upload_id": "uuid",
  "records": 1247,
  "analysis": {
    "hotspots": 12,
    "anomalies": 8,
    "trends": 12,
    "actors": 45
  }
}
```

### Upload Failed
```json
{
  "type": "upload_failed",
  "upload_id": "uuid",
  "error": "Error message"
}
```

## Performance

- **Small files (< 1000 rows):** 2-3 seconds
- **Medium files (1000-10000 rows):** 5-10 seconds
- **Large files (> 10000 rows):** 15-30 seconds

## Troubleshooting

### Upload Stuck at "processing"
- Check logs: `docker logs crisismap-backend`
- Verify MongoDB connection
- Check file size (max 50MB)

### Analysis Report Missing
- Ensure MongoDB is running
- Check database permissions
- Verify `analysis_reports` collection exists

### WebSocket Not Updating
- Check WebSocket connection
- Verify firewall allows WebSocket
- Check browser console for errors

## Rollback

If issues occur:
```bash
cp backend_v2/main_backup.py backend_v2/main.py
# Restart backend
```

## Support

For issues:
1. Check logs: `docker logs crisismap-backend`
2. Verify all dependencies are installed
3. Test with sample CSV file
4. Check MongoDB connection
