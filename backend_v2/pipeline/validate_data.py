import pandas as pd
from typing import List, Dict


def _required_columns_for_events() -> set:
    return {
        'event_id', 'event_date', 'year', 'event_type', 'sub_event_type',
        'actor1', 'actor2', 'region', 'country', 'admin1', 'admin2',
        'location', 'latitude', 'longitude', 'fatalities', 'source', 'notes',
    }


def _required_columns_for_indicators() -> set:
    return {
        'indicator_type', 'region', 'admin1', 'admin2', 'value', 'unit', 'source', 'date'
    }


def _required_columns_for_analysis_results() -> set:
    return {
        'id', 'analysis_type', 'country', 'result_data', 'created_at'
    }


def _infer_columns(df: pd.DataFrame) -> set:
    return set(df.columns.tolist()) if df is not None else set()


def validate_dataframe_schema(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, bool]]:
    """Validate that provided DataFrames have required schemas.

    dfs: mapping from logical name to DataFrame, e.g. {'events': df_events, 'analysis_results': df_ar}
    Returns a dict of per-DataFrame validation results: {name: {column: exists}}
    """
    results = {}
    if 'events' in dfs:
        cols = _infer_columns(dfs['events'])
        results['events'] = {c: (c in cols) for c in _required_columns_for_events()}
    if 'analysis_results' in dfs:
        cols = _infer_columns(dfs['analysis_results'])
        results['analysis_results'] = {c: (c in cols) for c in _required_columns_for_analysis_results()}
    if 'humanitarian_indicators' in dfs:
        cols = _infer_columns(dfs['humanitarian_indicators'])
        results['humanitarian_indicators'] = {c: (c in cols) for c in _required_columns_for_indicators()}
    return results


def validate_csv_paths(paths: List[str]) -> Dict[str, any]:
    """Load CSV/Excel files and validate their schemas. Returns summary dict."""
    dfs = {}
    for p in paths:
        if p.lower().endswith('.csv'):
            dfs_path = pd.read_csv(p, nrows=5)
        else:
            try:
                dfs_path = pd.read_excel(p, nrows=5)
            except Exception:
                dfs_path = None
        if isinstance(dfs_path, pd.DataFrame):
            # store a small representative preview as a DataFrame per path
            dfs[p] = dfs_path
    # If no valid DataFrames found, return empty
    if not dfs:
        return {"valid": False, "reason": "No valid CSV/Excel files found"}
    # Build a simple mapping for validation by filename
    validation = {}
    for path, df in dfs.items():
        validation[path] = _infer_columns(df)
    return {"valid": True, "validation": validation}


__all__ = [
"validate_dataframe_schema",
"validate_csv_paths",
]
