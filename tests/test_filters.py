import pandas as pd

from wa_analyzer.app.filters import apply_global_filters, is_number


def test_apply_global_filters_removes_channels_archives_and_non_contacts():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
            ),
            "message_type": [0, 0, 0, 7],
            "raw_string": [
                "12345@s.whatsapp.net",
                "friend@s.whatsapp.net",
                "status@broadcast",
                "ignored@s.whatsapp.net",
            ],
            "chat_row_id": [1, 2, 3, 4],
            "chat_name": ["12345", "Friend", "Status", "Ignored"],
            "contact_name": ["12345", "Friend", "Status", "Ignored"],
            "sender_string": ["12345@s.whatsapp.net"] * 4,
            "from_me": [0, 1, 0, 1],
        }
    )

    filtered, group_base = apply_global_filters(
        df,
        date_start=pd.Timestamp("2025-01-01").date(),
        date_end=pd.Timestamp("2025-01-31").date(),
        exclude_groups=False,
        exclude_archived=True,
        archived_chat_ids=frozenset({2}),
        exclude_low_participation=True,
        exclude_channels=True,
        exclude_family_global=False,
        family_tuple=(),
        exclude_non_contacts=True,
    )

    assert filtered.empty
    assert group_base.empty
    assert is_number("12345")
    assert not is_number("Friend")

