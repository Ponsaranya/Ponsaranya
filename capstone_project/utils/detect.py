
# import re

# def detect_module(text):
#     text_lower = text.lower()

#     parts = {
#         "faq": None,
#         "recommendation": None
#     }

#     if "suggest" in text_lower or "recommend" in text_lower:
#         split_words = re.split(r'suggest|recommend', text, flags=re.IGNORECASE)
#         parts["faq"] = split_words[0].strip()
#         parts["recommendation"] = "suggest " + split_words[1].strip()
#     else:
#         parts["faq"] = text.strip()

#     return parts

# detect.py
import re

def detect_module(text):
    original_text = text.strip()
    text_lower = original_text.lower()

    parts = {
        "faq": None,
        "recommendation": None
    }

    # Recommendation keywords
    rec_keywords = [
        "suggest",
        "recommend",
        "loan",
        "investment",
        "invest",
        "credit",
        "customer id",
        "savings",
        "saving"
    ]

    # Check if recommendation intent exists
    is_recommendation = any(k in text_lower for k in rec_keywords)

    if is_recommendation:
        # Extract customer ID if present
        cid_match = re.search(r'c\d{5,}', text_lower)

        # Keep recommendation text intact
        parts["recommendation"] = original_text

        # Remove recommendation-related words for FAQ
        faq_text = original_text
        faq_text = re.sub(r'suggest|recommend|loan|investment|invest|savings|saving|customer id|c\d+', '',
                          faq_text, flags=re.IGNORECASE)

        faq_text = faq_text.replace("and", "").strip()

        if faq_text:
            parts["faq"] = faq_text

    else:
        # Only FAQ
        parts["faq"] = original_text

    return parts