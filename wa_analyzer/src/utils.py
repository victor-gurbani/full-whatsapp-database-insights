import gender_guesser.detector as gender
import datetime
import os
import re
import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_user_timezone():
    """Return the timezone configured on the computer running the app."""
    tz_name = os.environ.get("TZ")
    if not tz_name and os.path.exists("/etc/localtime"):
        localtime_path = os.path.realpath("/etc/localtime")
        marker = "zoneinfo/"
        if marker in localtime_path:
            tz_name = localtime_path.split(marker, 1)[1]

    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            pass

    return datetime.datetime.now().astimezone().tzinfo


def to_user_datetime(values, unit="ms"):
    """Convert WhatsApp UTC timestamps to the computer's local wall time."""
    converted = pd.to_datetime(values, unit=unit, errors="coerce", utc=True)
    user_timezone = get_user_timezone()
    if isinstance(converted, pd.Series):
        return converted.dt.tz_convert(user_timezone).dt.tz_localize(None)
    if pd.isna(converted):
        return converted
    return converted.tz_convert(user_timezone).tz_localize(None)


class Utils:
    def __init__(self):
        self.detector = gender.Detector()
        # Manual Overrides for nicknames and specific terms
        self.overrides = {
            'javi': 'male',
            'javier': 'male',
            'rodri': 'male',
            'rodrigo': 'male',
            'abuela': 'female',
            'abuelo': 'male',
            'mama': 'female',
            'papa': 'male',
            'tio': 'male',
            'tia': 'female',
            'primo': 'male',
            'prima': 'female',
            'dani': 'male', # Often Daniel
            'alex': 'male', # Ambiguous but often Alexander/Alejandro
            'nacho': 'male',
            'pepe': 'male',
            'lola': 'female',
            'fran': 'male',
            'manu': 'male',
            'nico': 'male',
            'tom': 'male',
            'sam': 'male'
        }

    def guess_gender(self, name):
        if not name:
            return 'unknown'
        
        # 0. Handle Self (You/Me) - Default to Unknown or User Preference?
        # Typically "You" should not be counted in gender stats of contacts.
        # We classify as 'unknown' here, and Analyzer should filter 'from_me' rows.
        if name.lower() in ['you', 'me', 'myself', 'yo', 'tú']:
            return 'unknown'

        # 1. Clean up first name
        # Handle hyphens
        # We split by space OR hyphen first.
        name_clean = name.replace('-', ' ').replace('_', ' ')
        
        # Remove emojis/special chars (keep letters and spaces)
        name_clean = re.sub(r'[^\w\s]', '', name_clean)
        
        if not name_clean:
            return 'unknown'
            
        first_word = name_clean.split()[0]
        # Remove numbers if any (e.g. "Dad 2")
        first_word = re.sub(r'\d+', '', first_word)
        
        if not first_word or len(first_word) < 2:
            return 'unknown'
            
        lower_name = first_word.lower()
        
        # 2. Check Overrides
        if lower_name in self.overrides:
            return self.overrides[lower_name]
            
        # 3. Use Detector
        # Detector usually expects Title Case (e.g. "Jesus")
        g = self.detector.get_gender(first_word.capitalize())
        
        # Map detector output to simple categories
        if 'female' in g:
            return 'female'
        elif 'male' in g:
            return 'male'
        else:
            return 'unknown'

    def clean_text(self, text):
        if not text:
            return ""
        return str(text).lower()
