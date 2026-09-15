from __future__ import annotations

import urllib.parse
from typing import Iterable

from ..http import HTTPClient
from ..types import Candidate


LIST_URL = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
PREFIX = "nutrition5k_dataset/imagery/realsense_overhead/"
LANDING_URL = "https://github.com/google-research-datasets/Nutrition5k"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


class Nutrition5kSource:
    name = "nutrition5k"

    def __init__(self, http: HTTPClient) -> None:
        self.http = http

    @staticmethod
    def _is_rgb(name: str) -> bool:
        lower = name.lower()
        return lower.endswith(("rgb.png", "rgb.jpg", "rgb.jpeg")) and "depth" not in lower

    def discover(self, max_hint: int) -> Iterable[Candidate]:
        emitted = 0
        page_token: str | None = None
        while emitted < max_hint:
            params = {"prefix": PREFIX, "maxResults": "1000"}
            if page_token:
                params["pageToken"] = page_token
            payload = self.http.get_json(LIST_URL, params)
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", ""))
                if not self._is_rgb(name):
                    continue
                encoded = urllib.parse.quote(name, safe="/")
                dish_id = next((part for part in name.split("/") if part.startswith("dish_")), name)
                yield Candidate(
                    provider=self.name,
                    source_id=name,
                    download_url=f"https://storage.googleapis.com/nutrition5k_dataset/{encoded}",
                    landing_url=LANDING_URL,
                    license_name="CC BY 4.0",
                    license_url=LICENSE_URL,
                    creator="Quin Thames, Arjun Karpur, and the Nutrition5k authors",
                    title=f"Nutrition5k {dish_id} overhead RGB image",
                    attribution=(
                        f"Nutrition5k {dish_id}, Thames et al.; CC BY 4.0; {LANDING_URL}"
                    ),
                    extra={
                        "gcs_object": name,
                        "gcs_generation": item.get("generation"),
                        "gcs_updated": item.get("updated"),
                        "dataset_citation": "Thames et al., CVPR 2021, Nutrition5k",
                    },
                )
                emitted += 1
                if emitted >= max_hint:
                    return
            page_token = payload.get("nextPageToken")
            if not page_token:
                return
