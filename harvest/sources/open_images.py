from __future__ import annotations

import csv
import io
from typing import Iterable

from ..http import HTTPClient
from ..types import Candidate


CLASS_URL = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
LABEL_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv"
METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
MATCH_TERMS = (
    "food",
    "dish",
    "porridge",
    "congee",
    "soup",
    "drink",
    "tofu",
    "egg",
    "meat",
    "baby food",
)


class OpenImagesSource:
    """Use the small official validation split and preserve every image's metadata row."""

    name = "open-images"

    def __init__(self, http: HTTPClient) -> None:
        self.http = http

    def _matching_mids(self) -> set[str]:
        result: set[str] = set()
        for row in csv.reader(io.StringIO(self.http.get_text(CLASS_URL))):
            if len(row) < 2:
                continue
            label = row[1].casefold()
            if any(term in label for term in MATCH_TERMS):
                result.add(row[0])
        return result

    def discover(self, max_hint: int) -> Iterable[Candidate]:
        mids = self._matching_mids()
        if not mids:
            return
        labels_by_id: dict[str, set[str]] = {}
        label_reader = csv.DictReader(io.StringIO(self.http.get_text(LABEL_URL)))
        for row in label_reader:
            if row.get("LabelName") not in mids or row.get("Confidence") not in {"1", "1.0"}:
                continue
            labels_by_id.setdefault(row.get("ImageID", ""), set()).add(row["LabelName"])

        emitted = 0
        metadata_reader = csv.DictReader(io.StringIO(self.http.get_text(METADATA_URL)))
        for row in metadata_reader:
            image_id = row.get("ImageID", "")
            if image_id not in labels_by_id:
                continue
            download_url = row.get("Thumbnail300KURL") or row.get("OriginalURL") or ""
            landing_url = row.get("OriginalLandingURL") or ""
            license_url = row.get("License") or ""
            author = row.get("Author") or ""
            title = row.get("Title") or f"Open Images {image_id}"
            if not all((download_url, landing_url, license_url)):
                continue
            yield Candidate(
                provider=self.name,
                source_id=image_id,
                download_url=download_url,
                landing_url=landing_url,
                license_name="CC BY 2.0" if "/by/2.0" in license_url else "unverified",
                license_url=license_url,
                creator=author,
                title=title,
                attribution=f"{title} — {author}; CC BY 2.0; {landing_url}",
                verification_status="unverified",
                extra={
                    "open_images_subset": row.get("Subset"),
                    "open_images_label_mids": sorted(labels_by_id[image_id]),
                    "author_profile_url": row.get("AuthorProfileURL"),
                    "original_url": row.get("OriginalURL"),
                    "original_md5": row.get("OriginalMD5"),
                    "license_verification_caveat": (
                        "Open Images records CC BY 2.0 but asks reusers to verify each image at source."
                    ),
                    "license_verification_landing_url": landing_url,
                },
            )
            emitted += 1
            if emitted >= max_hint:
                return
