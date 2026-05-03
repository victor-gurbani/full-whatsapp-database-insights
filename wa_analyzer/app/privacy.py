"""Privacy and anonymization helpers for the Streamlit app."""

import hashlib
import random as _random
import string

import pandas as pd
import streamlit as st


# --- Anonymisation Helpers ---
def _anon_hash(name):
    """Full SHA-256 hex hash of a name."""
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()


def _anon_hash_cut(name):
    """SHA-256 hex hash truncated to 8 characters."""
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:8]


def _anon_random(name):
    """Deterministic random string: 6-10 alphanumeric chars + 2-4 digits. Seeded from name for stability."""
    seed = int(hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest(), 16)
    rng = _random.Random(seed)
    alpha_len = rng.randint(6, 10)
    digit_len = rng.randint(2, 4)
    letters = "".join(rng.choices(string.ascii_letters, k=alpha_len))
    digits = "".join(rng.choices(string.digits, k=digit_len))
    return letters + digits


@st.cache_data(show_spinner=False)
def build_anon_map(unique_names, mode):
    """
    Build a stable mapping {original_name → anonymised_name}.
    'You' / 'Me' are never anonymised.
    mode: 'off' | 'hash' | 'hash_cut' | 'random'
    """
    if mode == "off":
        return {}
    fn = {"hash": _anon_hash, "hash_cut": _anon_hash_cut, "random": _anon_random}[mode]
    me_names = {"you", "me", "myself", "yo", "tú"}
    return {
        n: (n if str(n).strip().lower() in me_names else fn(str(n)))
        for n in unique_names
        if pd.notna(n)
    }


def apply_anon_to_df(df, anon_map, cols=("contact_name", "chat_name", "subject")):
    """Apply anonymisation mapping to identity columns in-place and return df."""
    if not anon_map:
        return df

    def _anon_value(value):
        return anon_map.get(value, value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(_anon_value)

    if "reactions_list" in df.columns:
        def _anon_reactions(value):
            if not isinstance(value, list):
                return value
            anonymized = []
            for reaction_tuple in value:
                if isinstance(reaction_tuple, (list, tuple)) and len(reaction_tuple) >= 2:
                    anonymized.append((reaction_tuple[0], _anon_value(reaction_tuple[1])))
                else:
                    anonymized.append(reaction_tuple)
            return anonymized

        df["reactions_list"] = df["reactions_list"].map(_anon_reactions)

    if "mentions_name_list" in df.columns:
        def _anon_mention_names(value):
            if not isinstance(value, list):
                return value
            return [_anon_value(name) for name in value]

        df["mentions_name_list"] = df["mentions_name_list"].map(_anon_mention_names)

    if "mentions_pairs" in df.columns:
        def _anon_mention_pairs(value):
            if not isinstance(value, list):
                return value
            anonymized = []
            for mention_tuple in value:
                if isinstance(mention_tuple, (list, tuple)) and len(mention_tuple) >= 2:
                    anonymized.append((mention_tuple[0], _anon_value(mention_tuple[1])))
                else:
                    anonymized.append(mention_tuple)
            return anonymized

        df["mentions_pairs"] = df["mentions_pairs"].map(_anon_mention_pairs)

    return df


def collect_nested_identity_values(df):
    """Collect names embedded in list/tuple columns so anonymisation covers them."""
    values = set()

    def _add(value):
        if pd.notna(value):
            values.add(value)

    if "reactions_list" in df.columns:
        for reactions in df["reactions_list"].dropna():
            if not isinstance(reactions, list):
                continue
            for reaction_tuple in reactions:
                if isinstance(reaction_tuple, (list, tuple)) and len(reaction_tuple) >= 2:
                    _add(reaction_tuple[1])

    if "mentions_name_list" in df.columns:
        for mentions in df["mentions_name_list"].dropna():
            if isinstance(mentions, list):
                for name in mentions:
                    _add(name)

    if "mentions_pairs" in df.columns:
        for mentions in df["mentions_pairs"].dropna():
            if not isinstance(mentions, list):
                continue
            for mention_tuple in mentions:
                if isinstance(mention_tuple, (list, tuple)) and len(mention_tuple) >= 2:
                    _add(mention_tuple[1])

    return values


def av(value, anon_numbers=False, rng_seed=None):
    """
    Anonymise Value: if anon_numbers is True, replace numeric value with a random
    value of similar magnitude (±50%). If False, return as-is.
    For display only — does not affect underlying computations.
    """
    if not anon_numbers:
        return value
    if value is None or (isinstance(value, float) and (pd.isna(value) or value == 0)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v == 0:
        return value
    seed = int(abs(v * 1000)) if rng_seed is None else rng_seed
    rng = _random.Random(seed)
    factor = rng.uniform(0.5, 1.5)
    result = v * factor
    # Preserve integer type if original was int-like
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        return int(round(result))
    return result


