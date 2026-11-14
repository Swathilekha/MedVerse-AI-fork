import re


PATTERNS = {
    "hemoglobin": [
        r"(?:hemoglobin|haemoglobin|hb)\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    ],

    "wbc": [
        r"(?:wbc|wbc\s*count|white\s*blood\s*cell(?:\s*count)?)\s*[:=-]?\s*([\d,]+)"
    ],

    "glucose": [
        r"(?:blood\s*glucose|glucose|blood\s*sugar)\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    ],

    "cholesterol": [
        r"(?:total\s*cholesterol|cholesterol)\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    ],

    "spo2": [
        # Handles SpO2, Sp02 and common OCR variations
        r"(?:spo2|sp02|spo\s*2|oxygen\s*saturation|o2\s*saturation)\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    ],

    "heart_rate": [
        r"(?:heart\s*rate|pulse)\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    ]
}


def extract_clinical_values(text: str) -> dict:
    result = {}

    # Normalize common OCR variations
    normalized_text = text.replace("Sp02", "SpO2")
    normalized_text = normalized_text.replace("sp02", "spO2")

    for field, patterns in PATTERNS.items():

        for pattern in patterns:
            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE
            )

            if match:
                value = match.group(1).replace(",", "")

                try:
                    result[field] = (
                        float(value)
                        if "." in value
                        else int(value)
                    )
                except ValueError:
                    pass

                break

    return result