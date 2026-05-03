import pandas as pd

from wa_analyzer.app.privacy import (
    _anon_hash,
    _anon_hash_cut,
    _anon_random,
    apply_anon_to_df,
    build_anon_map,
    collect_nested_identity_values,
)


def test_anonymization_is_stable_and_covers_nested_identity_values():
    original = "Synthetic Person"
    assert _anon_hash(original) == _anon_hash(original)
    assert _anon_hash(original) != original
    assert len(_anon_hash_cut(original)) == 8
    assert _anon_random(original) == _anon_random(original)

    df = pd.DataFrame(
        {
            "contact_name": [original, "Me"],
            "chat_name": [original, "Me"],
            "subject": [None, None],
            "reactions_list": [[("👍", original)], []],
            "mentions_name_list": [[original], []],
            "mentions_pairs": [[("123", original)], []],
        }
    )
    names = collect_nested_identity_values(df)
    assert original in names

    anon_map = build_anon_map(frozenset([original, "Me", *names]), "hash_cut")
    anonymized = apply_anon_to_df(df.copy(), anon_map)

    assert anonymized.loc[0, "contact_name"] != original
    assert anonymized.loc[1, "contact_name"] == "Me"
    assert anonymized.loc[0, "reactions_list"][0][1] != original
    assert anonymized.loc[0, "mentions_name_list"][0] != original
    assert anonymized.loc[0, "mentions_pairs"][0][1] != original

