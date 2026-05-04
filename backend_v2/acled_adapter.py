import requests
import pandas as pd
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import set_key

logger = logging.getLogger(__name__)


class ACLEDAdapter:
    """
    Adapter for ACLED API.
    Handles OAuth authentication, automatic token refresh, ACLED events
    endpoint, and the CAST (Conflict Alert System) prediction endpoint.
    """

    def __init__(self):
        self.base_url = "https://acleddata.com/api/"
        self.auth_url = "https://acleddata.com/oauth/token"
        self.email = os.getenv("ACLED_EMAIL")
        self.password = os.getenv("ACLED_PASSWORD")
        self.access_token = os.getenv("ACLED_ACCESS_TOKEN")
        self.refresh_token = os.getenv("ACLED_REFRESH_TOKEN")
        self.client_id = "acled"

    # ------------------------------------------------------------------ #
    # Authentication helpers
    # ------------------------------------------------------------------ #

    def _find_env_path(self) -> Optional[str]:
        """Locate the nearest .env file (current dir or one level up)."""
        for candidate in [".env", os.path.join("..", ".env")]:
            if os.path.exists(candidate):
                return candidate
        return None

    def _persist_tokens(self, access_token: str, refresh_token: str):
        """Save new tokens to .env and update instance attributes."""
        self.access_token = access_token
        self.refresh_token = refresh_token
        env_path = self._find_env_path()
        if env_path:
            set_key(env_path, "ACLED_ACCESS_TOKEN", access_token)
            set_key(env_path, "ACLED_REFRESH_TOKEN", refresh_token)
            logger.info(f"ACLED tokens persisted to {env_path}")
        else:
            logger.warning("No .env file found; tokens not persisted to disk")

    def authenticate(self, email: str = None, password: str = None) -> bool:
        """
        Obtain a fresh access + refresh token pair using email/password.
        Falls back to ACLED_EMAIL / ACLED_PASSWORD from environment if
        parameters are not supplied.
        """
        email = email or self.email
        password = password or self.password

        if not email or not password:
            logger.error("ACLED_EMAIL or ACLED_PASSWORD not set; cannot authenticate")
            return False

        payload = {
            "username": email,
            "password": password,
            "grant_type": "password",
            "client_id": self.client_id,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(self.auth_url, headers=headers, data=payload, timeout=20)
            if response.status_code == 200:
                token_data = response.json()
                self._persist_tokens(
                    token_data["access_token"],
                    token_data["refresh_token"]
                )
                logger.info(f"ACLED authentication successful for {email}")
                return True
            else:
                logger.error(f"ACLED auth failed: {response.status_code} {response.text}")
                return False
        except Exception as exc:
            logger.error(f"Exception during ACLED authentication: {exc}")
            return False

    def refresh_access_token(self) -> bool:
        """Use refresh_token to obtain a new access token."""
        if not self.refresh_token:
            logger.warning("No refresh token; falling back to password auth")
            return self.authenticate()

        payload = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            "client_id": self.client_id,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(self.auth_url, headers=headers, data=payload, timeout=20)
            if response.status_code == 200:
                token_data = response.json()
                self._persist_tokens(
                    token_data["access_token"],
                    token_data["refresh_token"]
                )
                logger.info("ACLED access token refreshed successfully")
                return True
            else:
                logger.warning(
                    f"ACLED token refresh failed ({response.status_code}); "
                    "attempting full re-authentication..."
                )
                return self.authenticate()
        except Exception as exc:
            logger.error(f"Exception during ACLED token refresh: {exc}")
            return False

    def ensure_authenticated(self) -> bool:
        """
        Called at startup. Always does a fresh password-based auth so the
        token is guaranteed to be valid for the next 24 hours.
        Ignores any cached ACLED_ACCESS_TOKEN in .env.
        """
        ok = self.authenticate()
        if ok:
            # Also reload into instance so _get() uses the brand new token
            self.access_token = os.getenv("ACLED_ACCESS_TOKEN")
        return ok

    # ------------------------------------------------------------------ #
    # Internal request helper
    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Make an authenticated GET request to any ACLED endpoint.
        Automatically retries once after refreshing the token on 401.
        """
        if not self.access_token:
            logger.error("No ACLED access token; call authenticate() first")
            return []

        if "_format" not in params:
            params["_format"] = "json"

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 401:
                logger.info("ACLED token expired; attempting refresh...")
                if self.refresh_access_token():
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                else:
                    return []

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    return data.get("data", [])
                logger.error(f"ACLED API error: {data.get('message', 'Unknown error')}")
                return []

            logger.error(f"ACLED request failed: {response.status_code} {response.text[:200]}")
            return []

        except Exception as exc:
            logger.error(f"Exception during ACLED request: {exc}")
            return []

    # ------------------------------------------------------------------ #
    # ACLED Events endpoint
    # ------------------------------------------------------------------ #

    def fetch_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch a single page of data from the ACLED events endpoint."""
        return self._get("acled/read", dict(filters or {}))

    def fetch_paginated_data(
        self,
        filters: Dict[str, Any] = None,
        max_records: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ACLED events with automatic pagination until max_records is
        reached or the API has no more data to return.
        """
        all_data: List[Dict[str, Any]] = []
        page = 1
        page_size = min(5000, max_records)
        params = dict(filters or {})

        logger.info(f"Starting paginated ACLED fetch — filters: {params}, max: {max_records}")

        while len(all_data) < max_records:
            params["limit"] = page_size
            params["page"] = page
            batch = self._get("acled/read", params)
            if not batch:
                break
            all_data.extend(batch)
            logger.info(f"  Page {page}: {len(batch)} records (total {len(all_data)})")
            if len(batch) < page_size:
                break
            page += 1

        return all_data[:max_records]

    def standardize_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert ACLED API records to a clean, standardized DataFrame."""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if "latitude" in df.columns:
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        if "longitude" in df.columns:
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        if "fatalities" in df.columns:
            df["fatalities"] = (
                pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
            )
        if "event_date" in df.columns:
            df["event_date"] = (
                pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            )

        return df

    # ------------------------------------------------------------------ #
    # CAST endpoint
    # ------------------------------------------------------------------ #

    def fetch_cast_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Fetch data from the CAST (Conflict Alert System) prediction endpoint.
        Filters can include: country, admin1, month, year, total_forecast,
        battles_forecast, erv_forecast, vac_forecast, timestamp, etc.
        """
        return self._get("cast/read", dict(filters or {}))

    def fetch_cast_paginated(
        self,
        filters: Dict[str, Any] = None,
        max_records: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Paginated fetch from the CAST endpoint."""
        all_data: List[Dict[str, Any]] = []
        page = 1
        page_size = min(5000, max_records)
        params = dict(filters or {})

        logger.info(f"Starting paginated CAST fetch — filters: {params}, max: {max_records}")

        while len(all_data) < max_records:
            params["limit"] = page_size
            params["page"] = page
            batch = self._get("cast/read", params)
            if not batch:
                break
            all_data.extend(batch)
            logger.info(f"  CAST page {page}: {len(batch)} records (total {len(all_data)})")
            if len(batch) < page_size:
                break
            page += 1

        return all_data[:max_records]

    def standardize_cast_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert CAST API records to a clean DataFrame with correct types."""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        int_cols = [
            "year", "total_forecast", "battles_forecast", "erv_forecast",
            "vac_forecast", "total_observed", "battles_observed",
            "erv_observed", "vac_observed", "timestamp",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        str_cols = ["country", "admin1", "month"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df
