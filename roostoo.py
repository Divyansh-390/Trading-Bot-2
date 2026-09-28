import os
import time
import hmac
import hashlib
import requests

from pathlib import Path
from dotenv import load_dotenv

# BASE_DIR = Path(__file__).resolve().parent.parent
# load_dotenv(BASE_DIR / ".env", override=True)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

print("ENV PATH:", ENV_PATH)
print("ENV EXISTS:", ENV_PATH.exists())
print("API KEY LOADED:", bool(os.getenv("ROOSTOO_API_KEY")))
print("SECRET LOADED:", bool(os.getenv("ROOSTOO_SECRET_KEY")))
class RoostooClient:
    def __init__(self):
        self.base_url = "https://mock-api.roostoo.com"

        self.api_key = os.getenv("ROOSTOO_API_KEY")
        self.secret_key = os.getenv("ROOSTOO_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Roostoo API credentials not found")

        self.session = requests.Session()
        self.server_time_offset = 0

    def _timestamp(self):
        return str(int(time.time() * 1000) + self.server_time_offset)

    def _sign(self, params):
        params = params.copy()

        params["timestamp"] = self._timestamp()

        total_params = "&".join(
            f"{key}={params[key]}"
            for key in sorted(params)
        )

        signature = hmac.new(
            self.secret_key.encode(),
            total_params.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature
        }

        return headers, params, total_params

    def _get(self, endpoint, params=None, signed=False):
        url = self.base_url + endpoint

        if signed:
            headers, params, _ = self._sign(params or {})
        else:
            headers = {}
            params = params or {}

        response = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    def _post(self, endpoint, params=None):
        url = self.base_url + endpoint

        headers, params, total_params = self._sign(params or {})

        headers["Content-Type"] = "application/x-www-form-urlencoded"

        response = self.session.post(
            url,
            headers=headers,
            data=total_params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    def server_time(self):
        return self._get("/v3/serverTime")

    def sync_time(self):
        local_time = int(time.time() * 1000)

        response = self.server_time()

        server_time = response["ServerTime"]

        self.server_time_offset = server_time - local_time

        return server_time

    def exchange_info(self):
        return self._get("/v3/exchangeInfo")

    def ticker(self, pair=None):
        params = {}

        if pair:
            params["pair"] = pair

        params["timestamp"] = self._timestamp()

        return self._get("/v3/ticker", params, signed=False)

    def balance(self):
        return self._get(
            "/v3/balance",
            signed=True
        )

    def pending_count(self):
        return self._get(
            "/v3/pending_count",
            signed=True
        )

    def place_order(
        self,
        pair,
        side,
        quantity,
        order_type="MARKET",
        price=None
    ):
        params = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity)
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price")

            params["price"] = str(price)

        return self._post(
            "/v3/place_order",
            params
        )

    def query_order(
        self,
        order_id=None,
        pair=None,
        offset=None,
        limit=None,
        pending_only=None
    ):
        params = {}

        if order_id is not None:
            params["order_id"] = str(order_id)

        elif pair is not None:
            params["pair"] = pair

            if pending_only is not None:
                params["pending_only"] = (
                    "TRUE" if pending_only else "FALSE"
                )

            if offset is not None:
                params["offset"] = str(offset)

            if limit is not None:
                params["limit"] = str(limit)

        return self._post(
            "/v3/query_order",
            params
        )

    def cancel_order(
        self,
        order_id=None,
        pair=None
    ):
        params = {}

        if order_id is not None:
            params["order_id"] = str(order_id)

        elif pair is not None:
            params["pair"] = pair

        return self._post(
            "/v3/cancel_order",
            params
        )

    def short_open(
        self,
        pair,
        collateral,
        price=None
    ):
        params = {
            "pair": pair,
            "collateral": str(collateral)
        }

        if price is not None:
            params["order_type"] = "LIMIT"
            params["price"] = str(price)

        return self._post(
            "/v6/short_open",
            params
        )

    def short_close(
        self,
        pair,
        close_qty=None,
        close_pct=None
    ):
        params = {
            "pair": pair
        }

        if close_qty is not None:
            params["close_qty"] = str(close_qty)

        elif close_pct is not None:
            params["close_pct"] = str(close_pct)

        return self._post(
            "/v6/short_close",
            params
        )

    def short_positions(self):
        return self._get(
            "/v6/short_positions",
            signed=True
        )