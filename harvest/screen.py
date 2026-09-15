from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .types import PrivacyResult


class PrivacyScreener:
    """Conservative local face/OCR screening; the VLM provides a second check."""

    def __init__(self) -> None:
        self._cv2: Any | None = None
        self._cascade: Any | None = None
        try:
            import cv2

            self._cv2 = cv2
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if not cascade.empty():
                self._cascade = cascade
        except (ImportError, AttributeError):
            pass

        self._pytesseract: Any | None = None
        try:
            import pytesseract

            self._pytesseract = pytesseract
        except ImportError:
            pass

    def screen(self, image: Image.Image) -> PrivacyResult:
        array = np.asarray(image)
        contains_face = False
        face_engine = "unavailable"
        details: dict[str, Any] = {}
        if self._cascade is not None and self._cv2 is not None:
            gray = self._cv2.cvtColor(array, self._cv2.COLOR_RGB2GRAY)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
            contains_face = len(faces) > 0
            details["face_regions"] = len(faces)
            face_engine = "opencv-haar-frontalface"

        contains_text = False
        text_engine = "unavailable"
        tokens: list[str] = []
        if self._pytesseract is not None:
            try:
                output = self._pytesseract.image_to_data(
                    image,
                    output_type=self._pytesseract.Output.DICT,
                )
                for token, confidence in zip(output.get("text", []), output.get("conf", [])):
                    try:
                        confident = float(confidence) >= 60
                    except (TypeError, ValueError):
                        confident = False
                    clean = str(token).strip()
                    if confident and len(clean) >= 2 and any(char.isalnum() for char in clean):
                        tokens.append(clean[:80])
                contains_text = bool(tokens)
                text_engine = "tesseract"
            except Exception as exc:
                text_engine = f"tesseract-error:{type(exc).__name__}"

        if self._cv2 is not None:
            try:
                decoded, points, _ = self._cv2.QRCodeDetector().detectAndDecode(
                    self._cv2.cvtColor(array, self._cv2.COLOR_RGB2BGR)
                )
                if points is not None or decoded:
                    contains_text = True
                    details["qr_or_barcode"] = True
                    text_engine += "+opencv-qr"
            except Exception:
                pass
        details["ocr_tokens"] = tokens
        return PrivacyResult(
            contains_face=contains_face,
            contains_text=contains_text,
            face_engine=face_engine,
            text_engine=text_engine,
            details=details,
        )
