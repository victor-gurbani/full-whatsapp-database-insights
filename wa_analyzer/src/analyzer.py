import pandas as pd
import re
from .utils import Utils

# Custom Stopwords (Spanish + Common Protocols)
STOPWORDS = {
    # Protocols & Metadata
    "https",
    "http",
    "com",
    "www",
    "image",
    "omitted",
    "video",
    "audio",
    "sticker",
    "gif",
    # Spanish Common Words
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "se",
    "del",
    "las",
    "un",
    "por",
    "con",
    "una",
    "su",
    "para",
    "es",
    "al",
    "lo",
    "como",
    "mas",
    "o",
    "pero",
    "sus",
    "le",
    "ha",
    "me",
    "si",
    "sin",
    "sobre",
    "este",
    "ya",
    "entre",
    "cuando",
    "todo",
    "esta",
    "ser",
    "son",
    "dos",
    "tambien",
    "era",
    "muy",
    "anos",
    "hasta",
    "desde",
    "esta",
    "mi",
    "porque",
    "que",
    "solo",
    "han",
    "yo",
    "hay",
    "vez",
    "puede",
    "todos",
    "asi",
    "nos",
    "ni",
    "parte",
    "tiene",
    "eso",
    "uno",
    "donde",
    "bien",
    "tiempo",
    "mismo",
    "ese",
    "ahora",
    "cada",
    "e",
    "vida",
    "otro",
    "despues",
    "te",
    "otros",
    "aunque",
    "esa",
    "eso",
    "hace",
    "otra",
    "gobierno",
    "tan",
    "durante",
    "siempre",
    "dia",
    "tanto",
    "ella",
    "tres",
    "si",
    "gran",
    "pais",
    "segun",
    "menos",
    "mundo",
    "ano",
    "antes",
    "estado",
    "contra",
    "sino",
    "forma",
    "caso",
    "nada",
    "hacer",
    "general",
    "estaba",
    "poco",
    "estos",
    "presidente",
    "mayor",
    "ante",
    "unos",
    "les",
    "algo",
    "hacia",
    "casa",
    "ellos",
    "ayer",
    "hecho",
    "primera",
    "mucho",
    "mientras",
    "ademas",
    "quien",
    "momento",
    "millones",
    "esto",
    "espana",
    "hombre",
    "estan",
    "pues",
    "hoy",
    "lugar",
    "madrid",
    "nacional",
    "trabajo",
    "otras",
    "mejor",
    "nuevo",
    "decir",
    "algunos",
    "entonces",
    "todas",
    "dias",
    "debe",
    "politica",
    "como",
    "casi",
    "toda",
    "tal",
    "largo",
    "quiere",
    "fueron",
    "no",
}


class WhatsappAnalyzer:
    def __init__(self, data: pd.DataFrame, use_medians=False):
        self.data = data.copy()
        self.utils = Utils()
        self.use_medians = use_medians
        self.stat_func = "median" if use_medians else "mean"
        if "gender" not in self.data.columns:
            # Ensure 'contact_name' is string
            self.data["contact_name"] = (
                self.data["contact_name"].fillna("Unknown").astype(str)
            )
            self.data["gender"] = self.data["contact_name"].apply(
                self.utils.guess_gender
            )

        # Identify groups
        # Groups have raw_string ending in @g.us OR non-null subject (if parser provided it)
        # We rely on raw_string mostly.
        if "raw_string" in self.data.columns:
            self.data["is_group"] = (
                self.data["raw_string"].astype(str).str.endswith("@g.us")
            )
        else:
            self.data["is_group"] = False  # Fallback

    def get_basic_stats(self):
        stats = {}
        stats["total_messages"] = len(self.data)
        stats["unique_contacts"] = self.data["contact_name"].nunique()

        # Sent vs Received
        sent_count = self.data[self.data["from_me"] == 1].shape[0]
        received_count = self.data[self.data["from_me"] == 0].shape[0]
        stats["sent_count"] = sent_count
        stats["received_count"] = received_count

        return stats

    def get_top_talkers(self, n=20, metric="messages", exclude_me=False):
        """
        metric: 'messages' or 'words'
        Returns: DataFrame with 'contact_name' (actually Chat Name), 'count', 'gender', 'is_group'
        """
        df = self.data
        if exclude_me:
            df = df[df["from_me"] == 0]

        if metric == "words":
            # Calculate word count locally
            df_text = df[df["text_data"].notnull()].copy()
            df_text["wc"] = df_text["text_data"].astype(str).str.split().str.len()

            counts = (
                df_text.groupby("chat_name")["wc"]
                .sum()
                .sort_values(ascending=False)
                .head(n)
                .reset_index()
            )
            counts.columns = ["contact_name", "count"]
        else:
            # Return dataframe with count, gender, AND is_group
            counts = df["chat_name"].value_counts().head(n).reset_index()
            counts.columns = ["contact_name", "count"]

        # Mapping for efficiency (from Chat Name)
        # Note: Gender of a 'Chat' (1-on-1) is the gender of the person.
        # Gender of a Group is None/Unknown.
        meta_map = df.drop_duplicates("chat_name").set_index("chat_name")[
            ["gender", "is_group"]
        ]
        counts = counts.merge(
            meta_map, left_on="contact_name", right_index=True, how="left"
        )

        return counts

    def get_hourly_activity(self, split_by=None, exclude_me=False):
        """
        split_by: 'gender', 'group', 'sender', or None
        """
        df = self.data
        if exclude_me:
            df = df[df["from_me"] == 0]

        if split_by == "gender":
            return (
                df.groupby([df["timestamp"].dt.hour, "gender"])
                .size()
                .unstack(fill_value=0)
            )
        elif split_by == "group":
            df_temp = df.copy()
            df_temp["type"] = df_temp["is_group"].map(
                {True: "Group", False: "Individual"}
            )
            return (
                df_temp.groupby([df_temp["timestamp"].dt.hour, "type"])
                .size()
                .unstack(fill_value=0)
            )
        elif split_by == "sender":
            df_temp = df.copy()
            df_temp["sender"] = df_temp["from_me"].map({1: "Me", 0: "Them"})
            return (
                df_temp.groupby([df_temp["timestamp"].dt.hour, "sender"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            return df["timestamp"].dt.hour.value_counts().sort_index()

    def get_daily_activity(self):
        return self.data["timestamp"].dt.day_name().value_counts()

    def get_monthly_activity(self, split_by=None, exclude_me=False):
        """
        split_by: 'gender', 'group', 'sender', or None
        """
        df = self.data
        if exclude_me:
            df = df[df["from_me"] == 0]

        if split_by:
            # Default target column
            col = "gender" if split_by == "gender" else "is_group"
            target_df = df

            if split_by == "group":
                target_df = df.copy()
                target_df["type"] = target_df["is_group"].map(
                    {True: "Group", False: "Individual"}
                )
                col = "type"
            elif split_by == "sender":
                target_df = df.copy()
                target_df["sender"] = target_df["from_me"].map({1: "Me", 0: "Them"})
                col = "sender"

            return (
                target_df.set_index("timestamp")
                .groupby(col)
                .resample("ME", include_groups=False)
                .size()
                .unstack(level=0)
                .fillna(0)
            )
        else:
            return df.set_index("timestamp").resample("ME").size()

    def get_behavioral_timeline(self, ghost_thresh=86400, init_thresh=21600):
        """
        Returns monthly counts for:
        - Ghosted by Them (I sent last)
        - Ghosted by Me (They sent last)
        - Initiated by Me
        - Initiated by Them
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Calculate time gaps
        df["next_timestamp"] = df["timestamp"].shift(-1)
        df["time_to_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

        df["prev_timestamp"] = df["timestamp"].shift(1)
        df["time_from_prev"] = (
            df["timestamp"] - df["prev_timestamp"]
        ).dt.total_seconds()

        # Since we might be analyzing a single chat, the shift works fine.
        # If analyzing global, we'd need to group by jid, but for Chat Explorer this is fine.

        # Ghosting
        # Refined Ghosting Logic (Match usage in get_ghosting_stats)
        max_ts = self.data["timestamp"].max()

        # Calculate effective next timestamp
        # Logic: If next JID is different (or None), assume silence extends to NOW (max_ts)
        # unless user abandoned app. But for stats, it counts as silence.

        # We need next_jid for the same row comparison
        df["next_jid"] = df["jid_row_id"].shift(-1)

        same_chat = df["jid_row_id"] == df["next_jid"]

        effective_next_ts = df["next_timestamp"].copy()
        effective_next_ts.loc[~same_chat] = max_ts
        effective_next_ts.loc[df["next_jid"].isna()] = max_ts

        df["effective_gap"] = (effective_next_ts - df["timestamp"]).dt.total_seconds()

        ghosted_by_them = df[
            (df["from_me"] == 1) & (df["effective_gap"] > ghost_thresh)
        ]
        ghosted_by_me = df[(df["from_me"] == 0) & (df["effective_gap"] > ghost_thresh)]

        # Initiations (First msg or > thresh gap)
        initiations = df[
            (df["time_from_prev"] > init_thresh) | df["time_from_prev"].isna()
        ]
        init_me = initiations[initiations["from_me"] == 1]
        init_them = initiations[initiations["from_me"] == 0]

        # Resample
        timeline = pd.DataFrame()
        timeline["Ghosted by Them"] = (
            ghosted_by_them.set_index("timestamp").resample("ME").size()
        )
        timeline["Ghosted by Me"] = (
            ghosted_by_me.set_index("timestamp").resample("ME").size()
        )
        timeline["Initiated by Me"] = (
            init_me.set_index("timestamp").resample("ME").size()
        )
        timeline["Initiated by Them"] = (
            init_them.set_index("timestamp").resample("ME").size()
        )

        return timeline.fillna(0)

    def get_activity_over_time_by_contact(self, contact_list):
        """
        Returns monthly activity for specific chats.
        """
        df_filtered = self.data[self.data["chat_name"].isin(contact_list)]
        return (
            df_filtered.set_index("timestamp")
            .groupby("chat_name")
            .resample("ME", include_groups=False)
            .size()
            .unstack(level=0)
            .fillna(0)
        )

    def get_message_dispersion_over_time(
        self, split_by=None, exclude_me=False, freq="ME"
    ):
        """
        Calculates message dispersion over time using normalized Shannon entropy.

        Dispersion measures how evenly messages are spread across chats:
        - 100% = Messages equally distributed across all active chats
        - 0% = All messages go to a single chat

        split_by: 'gender', 'group', or None
        exclude_me: If True, only consider received messages (from_me=0)
        freq: Resample frequency (default 'ME' = month end)

        Returns: DataFrame with dispersion percentage per time period
        """
        import numpy as np

        df = self.data.copy()
        if exclude_me:
            df = df[df["from_me"] == 0]

        if df.empty:
            return pd.DataFrame()

        def calc_dispersion(group):
            """Calculate normalized Shannon entropy for a group of messages."""
            if group.empty:
                return float(
                    "nan"
                )  # No data for this period - will create gap in chart
            counts = group["chat_name"].value_counts()
            n = len(counts)
            if n <= 1:
                return float("nan")  # Only one chat - not meaningful dispersion

            # Proportions
            proportions = counts / counts.sum()

            # Shannon entropy: -sum(p * log(p))
            entropy = -np.sum(proportions * np.log(proportions))

            # Normalize by max entropy (log(n)) to get percentage
            max_entropy = np.log(n)
            normalized = (entropy / max_entropy) * 100

            return normalized

        # Set timestamp as index for resampling
        df = df.set_index("timestamp")

        if split_by is None:
            # Overall dispersion per period
            result = df.resample(freq).apply(calc_dispersion)
            return result

        elif split_by == "gender":
            # Dispersion per gender per period
            results = {}
            for gender in ["male", "female", "unknown"]:
                gender_df = df[df["gender"] == gender]
                if not gender_df.empty:
                    results[gender] = gender_df.resample(freq).apply(calc_dispersion)

            if not results:
                return pd.DataFrame()
            return pd.DataFrame(results)

        elif split_by == "group":
            # Dispersion for groups vs individuals
            results = {}
            for is_group, label in [(True, "Group"), (False, "Individual")]:
                type_df = df[df["is_group"] == is_group]
                if not type_df.empty:
                    results[label] = type_df.resample(freq).apply(calc_dispersion)

            if not results:
                return pd.DataFrame()
            return pd.DataFrame(results)

        else:
            return pd.DataFrame()

    def calculate_response_times(self):
        # Sort by chat and time
        df = self.data.sort_values(["jid_row_id", "timestamp"])

        # Calculate time difference
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

        # We want time taken for ME to reply to THEM
        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_jid"] = df["jid_row_id"].shift(1)

        reply_times = df[
            (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["jid_row_id"] == df["prev_jid"])
        ]["time_diff"]

        # Filter outliers (> 24h)
        agg_reply = getattr(reply_times[reply_times < 86400], self.stat_func)()
        return agg_reply / 60 if pd.notnull(agg_reply) else 0

    def calculate_chat_reply_times(self):
        """Calculates avg reply time for Me and Them in the current dataset"""
        df = self.data.sort_values(["timestamp"]).copy()
        df["read_at"] = pd.to_datetime(df["read_at"], errors="coerce")
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
        df["prev_from_me"] = df["from_me"].shift(1)

        # My Reply Time: Current is from Me, Prev is from Them
        my_replies = df[(df["from_me"] == 1) & (df["prev_from_me"] == 0)]
        my_agg = getattr(
            my_replies[my_replies["time_diff"] < 86400]["time_diff"], self.stat_func
        )()

        # Their Reply Time: Current is from Them, Prev is from Me
        their_replies = df[(df["from_me"] == 0) & (df["prev_from_me"] == 1)]
        their_agg = getattr(
            their_replies[their_replies["time_diff"] < 86400]["time_diff"],
            self.stat_func,
        )()

        return (my_agg / 60 if pd.notnull(my_agg) else 0), (
            their_agg / 60 if pd.notnull(their_agg) else 0
        )

    def calculate_reply_time_over_time(self, max_minutes=1440, freq="ME"):
        """
        Returns average reply time over time for Me and Them.
        max_minutes: ignore reply times above this threshold (default 24h).
        freq: resample frequency (default month-end).
        """
        df = self.data.sort_values(["timestamp"]).copy()

        if "message_type" in df.columns:
            df = df[df["message_type"] != 7]

        df["prev_from_me"] = df["from_me"].shift(1)
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

        max_seconds = max_minutes * 60

        # My Reply Time: Current is from Me, Prev is from Them
        my_replies = df[
            (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["time_diff"] >= 0)
            & (df["time_diff"] < max_seconds)
        ].copy()

        if not my_replies.empty:
            my_series = (
                (my_replies.set_index("timestamp")["time_diff"] / 60)
                .resample(freq)
                .apply(self.stat_func)
            )
        else:
            my_series = pd.Series(dtype="float64")

        # Their Reply Time: Current is from Them, Prev is from Me
        their_replies = df[
            (df["from_me"] == 0)
            & (df["prev_from_me"] == 1)
            & (df["time_diff"] >= 0)
            & (df["time_diff"] < max_seconds)
        ].copy()

        if not their_replies.empty:
            their_series = (
                (their_replies.set_index("timestamp")["time_diff"] / 60)
                .resample(freq)
                .apply(self.stat_func)
            )
        else:
            their_series = pd.Series(dtype="float64")

        if my_series.empty and their_series.empty:
            return pd.DataFrame()

        return pd.DataFrame({"Me": my_series, "Them": their_series}).dropna(how="all")

    def calculate_weekly_reply_time_stability(self, max_minutes=1440):
        """
        Calculates median and stability (IQR) of reply times grouped by week (freq="W-MON").
        Reply time = difference between LAST text from Person A and FIRST text from Person B.

        Returns: DataFrame with columns for each metric:
        - "Me_Median", "Me_IQR", "Me_MAD"
        - "Them_Median", "Them_IQR", "Them_MAD"
        Indexed by week (W-MON).
        """
        df = self.data.sort_values(["timestamp"]).copy()

        if "message_type" in df.columns:
            df = df[df["message_type"] != 7]

        df["prev_from_me"] = df["from_me"].shift(1)
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

        max_seconds = max_minutes * 60

        # My Reply Time: Current is from Me, Prev is from Them
        my_replies = df[
            (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["time_diff"] >= 0)
            & (df["time_diff"] < max_seconds)
        ].copy()

        # Their Reply Time: Current is from Them, Prev is from Me
        their_replies = df[
            (df["from_me"] == 0)
            & (df["prev_from_me"] == 1)
            & (df["time_diff"] >= 0)
            & (df["time_diff"] < max_seconds)
        ].copy()

        result_rows = []

        # Helper to compute median and stability metrics
        def compute_metrics(group_df):
            """Returns (median, q25, q75, iqr, mad) in minutes."""
            if group_df.empty:
                return None, None, None, None, None

            times_min = group_df["time_diff"] / 60.0

            median = times_min.median()

            # IQR
            q75 = times_min.quantile(0.75)
            q25 = times_min.quantile(0.25)
            iqr = q75 - q25

            # MAD (Median Absolute Deviation)
            mad = (times_min - median).abs().median()

            return median, q25, q75, iqr, mad

        # Group by week
        if not my_replies.empty:
            my_replies["week"] = my_replies["timestamp"].dt.to_period("W-MON")
            for week, group in my_replies.groupby("week"):
                median, q25, q75, iqr, mad = compute_metrics(group)
                if median is not None:
                    result_rows.append(
                        {
                            "week": week.to_timestamp(),
                            "Me_Median": median,
                            "Me_Q25": q25,
                            "Me_Q75": q75,
                            "Me_IQR": iqr,
                            "Me_MAD": mad,
                        }
                    )

        if not their_replies.empty:
            their_replies["week"] = their_replies["timestamp"].dt.to_period("W-MON")
            for week, group in their_replies.groupby("week"):
                median, q25, q75, iqr, mad = compute_metrics(group)
                if median is not None:
                    # Check if this week already exists in result_rows
                    existing = next(
                        (r for r in result_rows if r["week"] == week.to_timestamp()),
                        None,
                    )
                    if existing:
                        existing["Them_Median"] = median
                        existing["Them_Q25"] = q25
                        existing["Them_Q75"] = q75
                        existing["Them_IQR"] = iqr
                        existing["Them_MAD"] = mad
                    else:
                        result_rows.append(
                            {
                                "week": week.to_timestamp(),
                                "Them_Median": median,
                                "Them_Q25": q25,
                                "Them_Q75": q75,
                                "Them_IQR": iqr,
                                "Them_MAD": mad,
                            }
                        )

        if not result_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(result_rows)
        result_df = result_df.set_index("week").sort_index()

        # Fill NaN with None for clarity
        return result_df.fillna(pd.NA)

    def calculate_chat_write_times(self):
        """
        Calculates avg WRITE time (Read -> Send Reply).
        Different from Reply Time (Received -> Send Reply).
        Returns: (my_avg_min, their_avg_min)
        """
        if "read_at" not in self.data.columns:
            # Missing column -> Failed to parse receipts or old cache
            return None, None

        df = self.data.sort_values(["timestamp"]).copy()

        # Filter out System Messages (7) and maybe Headers to ensure adjacency reflects conversation flow
        if "message_type" in df.columns:
            # Exclude System (7), Revoked (15), GP2 (11, 12 etc) - Keep standard messages
            # Safest: Keep 0-6 (Text/Media), 13 (Gif), 16 (LiveLoc)
            df = df[df["message_type"] != 7]

        # We need previous message details
        # Shift 1 to get the message I am replying TO
        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_read_at"] = df["read_at"].shift(
            1
        )  # When the PREVIOUS message was read
        df["prev_timestamp"] = df["timestamp"].shift(1)

        # My Write Time:
        # I am replying (from_me=1) to Their Msg (prev_from_me=0).
        # Time = My Timestamp - When I read Their Msg (prev_read_at).
        # Note: 'read_at' on Incoming Message = When I read it.
        # This is often missing in DB.
        my_replies = df[
            (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["prev_read_at"].notnull())
        ].copy()

        if not my_replies.empty:
            my_replies["write_time"] = (
                my_replies["timestamp"] - my_replies["prev_read_at"]
            ).dt.total_seconds()

            # Relaxed Filter: Allow small negative skew (up to -60s) treated as 0 (Read immediately/concurrently)
            # Filter outliers (> 240 min)
            my_replies.loc[my_replies["write_time"].between(-60, 0), "write_time"] = 0

            valid_mask = (my_replies["write_time"] >= 0) & (
                my_replies["write_time"] < 14400
            )  # 240 min
            valid_replies = my_replies[valid_mask]

            if not valid_replies.empty:
                my_stat = getattr(valid_replies["write_time"], self.stat_func)()
            else:
                my_stat = None
        else:
            my_stat = None

        # Their Write Time:
        # They are replying (from_me=0) to My Msg (prev_from_me=1).
        # Time = Their Timestamp - When They read My Msg (prev_read_at).
        # Note: 'read_at' on Outgoing Message = When they read it.
        their_replies = df[
            (df["from_me"] == 0)
            & (df["prev_from_me"] == 1)
            & (df["prev_read_at"].notnull())
        ].copy()

        if not their_replies.empty:
            their_replies["write_time"] = (
                their_replies["timestamp"] - their_replies["prev_read_at"]
            ).dt.total_seconds()

            # CRITICAL FIX: Median write_time can be negative (e.g. -6 mins) due to notification replies
            # or lazy read receipts. Instead of dropping them, treat as 0 (Immediate/intertwined).
            their_replies.loc[their_replies["write_time"] < 0, "write_time"] = 0

            # Filter outliers (> 240 min)
            valid_mask = (their_replies["write_time"] >= 0) & (
                their_replies["write_time"] < 14400
            )
            valid_replies = their_replies[valid_mask]

            if not valid_replies.empty:
                their_stat = getattr(valid_replies["write_time"], self.stat_func)()
            else:
                their_stat = None
        else:
            their_stat = None

        return (my_stat / 60 if pd.notnull(my_stat) else None), (
            their_stat / 60 if pd.notnull(their_stat) else None
        )

    def calculate_write_time_over_time(self, max_minutes=240, freq="ME"):
        """
        Returns average write time over time for Me and Them.
        max_minutes: ignore write times above this threshold.
        freq: resample frequency (default month-end).
        """
        if "read_at" not in self.data.columns:
            return pd.DataFrame()

        df = self.data.sort_values(["timestamp"]).copy()
        df["read_at"] = pd.to_datetime(df["read_at"], errors="coerce")

        if "message_type" in df.columns:
            df = df[df["message_type"] != 7]

        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_read_at"] = df["read_at"].shift(1)

        max_seconds = max_minutes * 60

        my_replies = df[
            (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["prev_read_at"].notnull())
        ].copy()

        their_replies = df[
            (df["from_me"] == 0)
            & (df["prev_from_me"] == 1)
            & (df["prev_read_at"].notnull())
        ].copy()

        for replies in (my_replies, their_replies):
            if not replies.empty:
                replies["write_time"] = (
                    replies["timestamp"] - replies["prev_read_at"]
                ).dt.total_seconds()
                replies.loc[replies["write_time"] < 0, "write_time"] = 0
                replies = replies[
                    (replies["write_time"] >= 0) & (replies["write_time"] < max_seconds)
                ]

        if not my_replies.empty:
            my_replies = my_replies[
                (my_replies["write_time"] >= 0)
                & (my_replies["write_time"] < max_seconds)
            ]
            my_series = (
                (my_replies.set_index("timestamp")["write_time"] / 60)
                .resample(freq)
                .apply(self.stat_func)
            )
        else:
            my_series = pd.Series(dtype="float64")

        if not their_replies.empty:
            their_replies = their_replies[
                (their_replies["write_time"] >= 0)
                & (their_replies["write_time"] < max_seconds)
            ]
            their_series = (
                (their_replies.set_index("timestamp")["write_time"] / 60)
                .resample(freq)
                .apply(self.stat_func)
            )
        else:
            their_series = pd.Series(dtype="float64")

        if my_series.empty and their_series.empty:
            return pd.DataFrame()

        return pd.DataFrame({"Me": my_series, "Them": their_series}).dropna(how="all")

    def calculate_gender_stats(self):
        metrics = []
        # Filter 'Me' out from gender stats
        # (Compare Men vs Women contacts, not Me vs Them where Me skews it)
        df_contacts = self.data[self.data["from_me"] == 0]

        for g in ["male", "female"]:
            g_data = df_contacts[df_contacts["gender"] == g]
            if g_data.empty:
                continue

            stats = {"gender": g}
            stats["count"] = len(g_data)

            # Avg WPM
            text_msgs = g_data[g_data["text_data"].notnull()]["text_data"].astype(str)
            if not text_msgs.empty:
                word_counts = text_msgs.apply(lambda x: len(x.split()))
                stats["avg_wpm"] = getattr(word_counts, self.stat_func)()
            else:
                stats["avg_wpm"] = 0

            # Reply Time
            # Logic: Calculate time for ME (from_me=1) to reply to THEM (from_me=0).
            # The 'gender' col in 'df_sorted' for MY messages is 'unknown'.
            # We verify the gender of the PREVIOUS SENDER ('prev_jid').

            # Map jid_row_id -> gender (using ONLY 'Them' messages, otherwise we get 'unknown' from 'Me' rows)
            them_data = self.data[self.data["from_me"] == 0]
            jid_gender_map = (
                them_data[["jid_row_id", "gender"]]
                .dropna(subset=["jid_row_id"])
                .drop_duplicates("jid_row_id")
                .set_index("jid_row_id")["gender"]
            )

            df_sorted = self.data.sort_values(["jid_row_id", "timestamp"]).copy()
            df_sorted["time_diff"] = df_sorted["timestamp"].diff().dt.total_seconds()
            df_sorted["prev_from_me"] = df_sorted["from_me"].shift(1)
            df_sorted["prev_jid"] = df_sorted["jid_row_id"].shift(1)

            # Add partner gender to calculated rows (by mapping prev_jid)
            df_sorted["partner_gender"] = df_sorted["prev_jid"].map(jid_gender_map)

            replies = df_sorted[
                (df_sorted["from_me"] == 1)
                & (df_sorted["prev_from_me"] == 0)
                & (df_sorted["jid_row_id"] == df_sorted["prev_jid"])
                & (df_sorted["partner_gender"] == g)
            ]

            valid_times = replies["time_diff"][replies["time_diff"] < 86400]
            stats["avg_reply_time"] = (
                (getattr(valid_times, self.stat_func)() / 60)
                if not valid_times.empty
                else 0
            )

            # Media Percentage
            if "mime_type" in g_data.columns:
                media_count = g_data["mime_type"].notnull().sum()
                stats["media_pct"] = (media_count / len(g_data)) * 100
            else:
                stats["media_pct"] = 0

            metrics.append(stats)

        return pd.DataFrame(metrics)

    def analyze_by_gender(self):
        # Exclude 'Me' from the pie chart
        return self.data[self.data["from_me"] == 0].groupby("gender").size()

    def get_ghosting_stats(self, threshold_seconds=86400, exclude_list=None):
        """
        Identify who leaves you on read.
        Threshold: 24h (86400) or 5 days (432000)
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Filter exclusions
        if exclude_list:
            df = df[~df["chat_name"].isin(exclude_list)]

        df["next_from_me"] = df["from_me"].shift(-1)
        df["next_jid"] = df["jid_row_id"].shift(-1)
        df["next_timestamp"] = df["timestamp"].shift(-1)

        df["time_to_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

        my_msgs = df[df["from_me"] == 1]

        # Define max time to check against end of data
        max_ts = self.data["timestamp"].max()

        # Logic:
        # 1. Next msg is different chat or missing (End of Chat Block) -> Check time vs Max Date
        # 2. Next msg is same chat -> Check time difference

        # Time to "Next Event" (Next Msg or End of Data)
        # If next_jid != current or next is na, effective time diff is max_ts - current

        # Vectorized approach:
        # Create 'effective_next_ts'
        # If same chat, use next_timestamp. Else use max_ts.
        same_chat = my_msgs["jid_row_id"] == my_msgs["next_jid"]

        effective_next_ts = my_msgs["next_timestamp"].copy()
        effective_next_ts.loc[~same_chat] = max_ts
        effective_next_ts.loc[my_msgs["next_jid"].isna()] = max_ts

        time_diff = (effective_next_ts - my_msgs["timestamp"]).dt.total_seconds()

        ghosted = my_msgs[time_diff > threshold_seconds]

        # Use chat_name to identify WHO is ghosting (the chat/person I'm talking to)
        # because for my messages, contact_name is 'You'.
        counts = ghosted["chat_name"].value_counts().head(20).reset_index()
        counts.columns = ["contact_name", "count"]

        # Add gender
        # We need to map chat_name -> gender using 'Them' (from_me=0) to avoid 'unknown' from Me.
        them_df = self.data[self.data["from_me"] == 0]
        gender_map = them_df.drop_duplicates("chat_name").set_index("chat_name")[
            "gender"
        ]
        counts["gender"] = counts["contact_name"].map(gender_map)

        return counts

    def get_left_on_read_stats(self, threshold_seconds=86400, exclude_list=None):
        """
        Identify who YOU leave on read.
        Refined Logic:
        - True Ghost: I read it (read_at present) but didn't reply > threshold.
        - Left on Delivered: I didn't read it (read_at is null) and didn't reply > threshold.
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Filter exclusions
        if exclude_list:
            df = df[~df["chat_name"].isin(exclude_list)]

        df["next_from_me"] = df["from_me"].shift(-1)
        df["next_jid"] = df["jid_row_id"].shift(-1)
        df["next_timestamp"] = df["timestamp"].shift(-1)

        df["time_to_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

        # Their Messages (From Them)
        their_msgs = df[df["from_me"] == 0]

        # Ignored Condition: No Reply OR Delayed Reply
        # Note: Original Ghosting logic considered "Delayed Reply" as ghosting too if > threshold.
        # "Left on Delivered" implies "Unopened".
        ignored = their_msgs[
            (their_msgs["next_jid"] != their_msgs["jid_row_id"])
            | (their_msgs["time_to_next"].isna())
            | (their_msgs["time_to_next"] > threshold_seconds)
        ]

        if ignored.empty:
            return pd.DataFrame()

        if ignored.empty:
            return pd.DataFrame()

        ignored = ignored.copy()

        # Robust Read Detection: Check 'read_at' AND 'status'
        # If read_at is missing, maybe status=13 (Blue Check) is present?
        if "read_at" not in ignored.columns:
            ignored["read_at"] = pd.NaT

        if "status" in ignored.columns:
            # status 13 = Read. If read_at is empty/NaT but status is 13, mark it as read.
            # We use timestamp as a proxy for read time just to make it non-null.
            mask_status_read = (ignored["status"] == 13) & (ignored["read_at"].isna())
            ignored.loc[mask_status_read, "read_at"] = ignored.loc[
                mask_status_read, "timestamp"
            ]

        # Classify
        # If read_at is not null -> True Ghost 👻
        # If read_at is null -> Left on Delivered 📨
        ignored.loc[:, "type"] = ignored["read_at"].apply(
            lambda x: "True Ghost 👻" if pd.notnull(x) else "Left on Delivered 📨"
        )

        stats = ignored.groupby(["contact_name", "type"]).size().unstack(fill_value=0)

        # Add Totals
        stats["Total Ignored"] = stats.sum(axis=1)

        # Add Gender
        gender_map = self.data.drop_duplicates("contact_name").set_index(
            "contact_name"
        )["gender"]
        stats["gender"] = stats.index.map(gender_map)

        return stats.sort_values("Total Ignored", ascending=False)

    def get_initiation_stats(self, threshold_seconds=21600, exclude_list=None):
        """
        Who starts conversations?
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Filter exclusions
        if exclude_list:
            df = df[~df["chat_name"].isin(exclude_list)]

        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
        df["prev_jid"] = df["jid_row_id"].shift(1)

        if "is_quote" not in df.columns:
            df["is_quote"] = False

        initiations = df[
            (df["jid_row_id"] != df["prev_jid"])
            | ((df["time_diff"] > threshold_seconds) & (df["is_quote"] == False))
        ].copy()

        their_initiations = initiations[initiations["from_me"] == 0][
            "contact_name"
        ].value_counts()
        # For ME, 'contact_name' is 'You', so we group by 'chat_name' to know WHO I initiated with.
        my_initiations = initiations[initiations["from_me"] == 1][
            "chat_name"
        ].value_counts()

        stats = pd.DataFrame({"Them": their_initiations, "Me": my_initiations}).fillna(
            0
        )
        stats["Total"] = stats["Them"] + stats["Me"]
        stats["% Them"] = (stats["Them"] / stats["Total"]) * 100

        return stats.sort_values("Total", ascending=False).head(20)

    def get_first_message_stats(self, exclude_list=None):
        """
        Who sent the very FIRST message in each chat? (Who broke the ice)
        Returns: (started_by_me, started_by_them, total_chats)
        """
        df = self.data.copy()

        # Filter exclusions
        if exclude_list:
            df = df[~df["chat_name"].isin(exclude_list)]

        # Get the first message per chat (by jid_row_id or chat_name)
        first_msgs = (
            df.sort_values("timestamp").groupby("chat_name").first().reset_index()
        )

        started_by_me = (first_msgs["from_me"] == 1).sum()
        started_by_them = (first_msgs["from_me"] == 0).sum()
        total_chats = len(first_msgs)

        return started_by_me, started_by_them, total_chats

    def get_reply_time_ranking(
        self, min_messages=20, max_delay_seconds=43200, exclude_list=None
    ):
        """
        Calculates average reply time per contact.
        Returns DataFrame with cols: contact_name, my_avg, their_avg (in minutes)
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Calculate deltas
        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_jid"] = df["jid_row_id"].shift(1)
        df["prev_timestamp"] = df["timestamp"].shift(1)

        df["time_delta"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()

        # Filter for valid response pairs (Same JID, Different Sender, Delta < Limit)
        # Limit to max_delay_seconds to exclude "new conversations" ghosting
        valid_responses = df[
            (df["jid_row_id"] == df["prev_jid"])
            & (df["from_me"] != df["prev_from_me"])
            & (df["time_delta"] < max_delay_seconds)
        ]

        # Group by CHAT (not contact, because contact is 'You' for me)
        stats = (
            valid_responses.groupby(["chat_name", "from_me"])["time_delta"]
            .apply(lambda x: getattr(x, self.stat_func)())
            .reset_index()
        )

        # Exclude list filter
        if exclude_list:
            stats = stats[~stats["chat_name"].isin(exclude_list)]

        # Valid contacts filter (enough messages total in the CHAT)
        # Use chat_name for correct count
        msg_counts = self.data["chat_name"].value_counts()
        valid_contacts = msg_counts[msg_counts >= min_messages].index

        stats = stats[stats["chat_name"].isin(valid_contacts)]

        # Pivot on chat_name
        # Pivot on chat_name
        stats_pivot = stats.pivot(
            index="chat_name", columns="from_me", values="time_delta"
        )

        # Ensure distinct columns [0, 1] exist
        stats_pivot = stats_pivot.reindex(columns=[0, 1])
        stats_pivot.columns = ["their_avg", "my_avg"]  # 0=Them, 1=Me

        # Convert seconds to minutes
        stats_pivot = stats_pivot / 60

        # Merge gender for coloring
        them_df = self.data[self.data["from_me"] == 0]
        gender_map = them_df.drop_duplicates("chat_name").set_index("chat_name")[
            "gender"
        ]
        stats_pivot["gender"] = stats_pivot.index.map(gender_map)

        # Reset index to make chat_name a column, rename to contact_name for UI checks
        stats_pivot = stats_pivot.reset_index().rename(
            columns={"chat_name": "contact_name"}
        )

        return stats_pivot.dropna().reset_index()

    def get_write_time_ranking(
        self, min_messages=25, max_delay_seconds=10800, exclude_list=None
    ):
        """
        Calculates average write time per contact (Read -> Send).
        Threshold: max_delay_seconds (default 180 min).
        Returns DataFrame with cols: contact_name, my_avg, their_avg (in minutes)
        """
        if "read_at" not in self.data.columns:
            return pd.DataFrame()

        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Exclude System messages
        if "message_type" in df.columns:
            df = df[df["message_type"] != 7]

        # Shift to get adjacency
        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_jid"] = df["jid_row_id"].shift(1)
        df["prev_read_at"] = df["read_at"].shift(1)

        # My Write Time: current from_me=1, prev from_me=0 (Their msg), same jid, prev_read_at exists
        my_mask = (
            (df["jid_row_id"] == df["prev_jid"])
            & (df["from_me"] == 1)
            & (df["prev_from_me"] == 0)
            & (df["prev_read_at"].notnull())
        )
        my_replies = df[my_mask].copy()

        # Their Write Time: current from_me=0, prev from_me=1 (My msg), same jid, prev_read_at exists
        their_mask = (
            (df["jid_row_id"] == df["prev_jid"])
            & (df["from_me"] == 0)
            & (df["prev_from_me"] == 1)
            & (df["prev_read_at"].notnull())
        )
        their_replies = df[their_mask].copy()

        # Calculate and Sanitize
        for d in [my_replies, their_replies]:
            if not d.empty:
                d["write_time"] = (
                    d["timestamp"] - d["prev_read_at"]
                ).dt.total_seconds()
                d.loc[d["write_time"].between(-60, 0), "write_time"] = (
                    0  # Handle race conditions
                )
                # Filter out > 180 min (considered new sessions or idle)
                d.drop(
                    d[~d["write_time"].between(0, max_delay_seconds)].index,
                    inplace=True,
                )

        if not my_replies.empty and "write_time" in my_replies.columns:
            my_stats = (
                my_replies.groupby("chat_name")["write_time"]
                .apply(lambda x: getattr(x, self.stat_func)())
                .reset_index(name="my_avg")
            )
        else:
            my_stats = pd.DataFrame(columns=["chat_name", "my_avg"])

        if not their_replies.empty and "write_time" in their_replies.columns:
            their_stats = (
                their_replies.groupby("chat_name")["write_time"]
                .apply(lambda x: getattr(x, self.stat_func)())
                .reset_index(name="their_avg")
            )
        else:
            their_stats = pd.DataFrame(columns=["chat_name", "their_avg"])

        stats = pd.merge(my_stats, their_stats, on="chat_name", how="outer")

        if exclude_list:
            stats = stats[~stats["chat_name"].isin(exclude_list)]

        # Min message filter (total in chat)
        msg_counts = self.data["chat_name"].value_counts()
        valid_contacts = msg_counts[msg_counts >= min_messages].index
        stats = stats[stats["chat_name"].isin(valid_contacts)]

        # Seconds to Minutes
        stats["my_avg"] = stats["my_avg"] / 60
        stats["their_avg"] = stats["their_avg"] / 60

        # Add gender
        them_df = self.data[self.data["from_me"] == 0]
        gender_map = them_df.drop_duplicates("chat_name").set_index("chat_name")[
            "gender"
        ]
        stats["gender"] = stats["chat_name"].map(gender_map)

        return stats.rename(columns={"chat_name": "contact_name"}).dropna(
            subset=["my_avg", "their_avg"], how="all"
        )

    def get_wordcloud_text(
        self,
        contact_name=None,
        filter_from_me=None,
        min_word_length=0,
        exclude_emails=False,
    ):
        """
        filter_from_me: True (only My words), False (only Their words), None (All)
        """
        if contact_name:
            df = self.data[self.data["contact_name"] == contact_name]
        else:
            df = self.data

        if filter_from_me is not None:
            v_from = 1 if filter_from_me else 0
            df = df[df["from_me"] == v_from]

        text_data = df["text_data"].dropna().astype(str)

        # Optimization: Join all text first, then apply regex once.
        # This is significantly faster than using .apply() on every row.
        full_text = " ".join(text_data.tolist())

        # Remove URLs (http/https and www.)
        full_text = re.sub(r"http\S+|www\.\S+", "", full_text)

        # Remove Emails if requested
        if exclude_emails:
            full_text = re.sub(r"\S+@\S+", "", full_text)

        words = full_text.lower().split()
        # Filter stopwords and short words
        valid_words = [
            w for w in words if w not in STOPWORDS and len(w) >= min_word_length
        ]

        return " ".join(valid_words)

    def get_behavioral_scorecard(self, exclude_groups=False):
        """
        Returns a DataFrame with behavioral metrics per contact:
        - night_owl_pct: % msgs sent 00:00-06:00
        - early_bird_pct: % msgs sent 06:00-09:00
        - double_text_ratio: "Consecutive turns" / "Total turns" ratio
        """
        if self.data.empty:
            return pd.DataFrame()

        df = self.data.copy()
        if exclude_groups:
            df = df[~df["is_group"]]

        # 1. Time based Metrics (Exclude Me)
        # We only care about when THEY message.
        df_bhv = df[df["from_me"] == 0].copy()

        df_bhv["hour"] = df_bhv["timestamp"].dt.hour
        total_counts = df_bhv["contact_name"].value_counts()

        night_counts = df_bhv[(df_bhv["hour"] >= 0) & (df_bhv["hour"] < 6)][
            "contact_name"
        ].value_counts()
        early_counts = df_bhv[(df_bhv["hour"] >= 6) & (df_bhv["hour"] < 9)][
            "contact_name"
        ].value_counts()

        stats = pd.DataFrame(
            {
                "total": total_counts,
                "night_owl": night_counts,
                "early_bird": early_counts,
            }
        ).fillna(0)

        stats["night_owl_pct"] = (stats["night_owl"] / stats["total"]) * 100
        stats["early_bird_pct"] = (stats["early_bird"] / stats["total"]) * 100

        # 2. Double Text Ratio (Refined: Turns with >1 message)
        # Old logic ("Resumed Monologue") yielded 99%.
        # New Logic: A "Turn" is a sequence of consecutive messages by same sender.
        # Ratio = (Turns with > 1 message) / (Total Turns)

        df_sorted = df.sort_values(["jid_row_id", "timestamp"])

        # Identify changes in sender OR chat (JID)
        # Shift to compare with previous
        df_sorted["prev_sender"] = df_sorted["contact_name"].shift(1)
        df_sorted["prev_jid"] = df_sorted["jid_row_id"].shift(1)

        # A new turn starts if Sender changes OR JID changes
        # Strict "Conversation Turn" definition
        change_mask = (df_sorted["contact_name"] != df_sorted["prev_sender"]) | (
            df_sorted["jid_row_id"] != df_sorted["prev_jid"]
        )

        # Assign Turn ID
        df_sorted["turn_id"] = change_mask.cumsum()

        # Aggregation: Count msgs per turn
        # Group by Turn ID -> (Contact, Count)
        turn_stats = (
            df_sorted.groupby(["turn_id", "contact_name"])
            .size()
            .reset_index(name="msg_count")
        )

        # Now analyze per contact
        # Total Turns
        total_turns = turn_stats["contact_name"].value_counts()

        # Double Text Turns (count > 1)
        multi_msg_turns = turn_stats[turn_stats["msg_count"] > 1][
            "contact_name"
        ].value_counts()

        stats["total_turns"] = total_turns
        stats["double_texts"] = multi_msg_turns
        stats = stats.fillna(0)

        stats["double_text_ratio"] = 0.0
        mask_turns = stats["total_turns"] > 5
        stats.loc[mask_turns, "double_text_ratio"] = (
            stats.loc[mask_turns, "double_texts"] / stats.loc[mask_turns, "total_turns"]
        ) * 100

        # Add Gender
        gender_map = self.data.drop_duplicates("contact_name").set_index(
            "contact_name"
        )["gender"]
        stats["gender"] = stats.index.map(gender_map)

        return stats

    def get_fun_stats(self, top_n=100, exclude_groups=False):
        """
        Returns stats for:
        - Laughs
        - Emojis (Top used - placeholder logic)
        - Media counts
        - Review/Deleted counts
        - Dry Texter (Avg Length)

        top_n: Filter analysis to top N contacts by message volume (default 100)
        exclude_groups: If True, remove group chats from analysis.
        """
        df = self.data.copy()

        # 0. Global Filters
        if exclude_groups:
            df = df[~df["is_group"]]

        # Filter out 'Me' (We want stats about contacts, not user)
        df = df[df["from_me"] == 0]

        # Filter for Top N Contacts (by volume) to keep stats relevant
        if top_n and top_n > 0:
            # Recalculate top contacts AFTER filtering groups/me
            top_contacts = df["contact_name"].value_counts().head(top_n).index
            df_top = df[df["contact_name"].isin(top_contacts)].copy()
        else:
            df_top = df

        # 1. Laughs
        laugh_pattern = r"(?i)(?:haha|jaja|lol|lmao|risas|xD|😂|🤣)"
        df_top["has_laugh"] = (
            df_top["text_data"].astype(str).str.contains(laugh_pattern, regex=True)
        )
        laugh_counts = df_top[df_top["has_laugh"]]["contact_name"].value_counts()

        # 2. Media & Deleted
        revoke_pattern = r"(?i)(?:This message was deleted|Eliminaste este mensaje|Se eliminó este mensaje)"
        is_revoked = df_top["text_data"].astype(str).str.fullmatch(revoke_pattern)
        if "message_type" in df_top.columns:
            is_revoked = is_revoked | (df_top["message_type"] == 15)

        deleted_counts = df_top[is_revoked]["contact_name"].value_counts()

        # Media
        media_mask = df_top["mime_type"].notnull()
        media_counts = df_top[media_mask]["contact_name"].value_counts()

        # 3. Dry Texter (Avg Words)
        # Filter for TEXT messages only (no media, no revokes)
        text_only = df_top[
            ~media_mask & ~is_revoked & df_top["text_data"].notnull()
        ].copy()
        text_only["word_count"] = (
            text_only["text_data"].astype(str).str.split().str.len()
        )

        # Filter 0 word length
        text_only = text_only[text_only["word_count"] > 0]

        agg_len = text_only.groupby("contact_name")["word_count"].apply(
            lambda x: getattr(x, self.stat_func)()
        )
        agg_len = agg_len[agg_len > 0]

        stats = pd.DataFrame(
            {
                "laughs": laugh_counts,
                "media": media_counts,
                "deleted": deleted_counts,
                "avg_word_len": agg_len,
            }
        ).fillna(0)

        # Add gender
        gender_map = self.data.drop_duplicates("contact_name").set_index(
            "contact_name"
        )["gender"]
        stats["gender"] = stats.index.map(gender_map)

        # 4. Detailed Media Breakdown
        # MIME types: audio/*, image/*, video/*, image/webp (sticker)
        if "mime_type" in df_top.columns:
            df_top["mime_cat"] = df_top["mime_type"].apply(
                lambda x: (
                    "audio"
                    if pd.notnull(x) and "audio" in x
                    else "image"
                    if pd.notnull(x) and "image" in x and "webp" not in x
                    else "video"
                    if pd.notnull(x) and "video" in x
                    else "sticker"
                    if pd.notnull(x) and ("webp" in x or "sticker" in x)
                    else "other"
                    if pd.notnull(x)
                    else None
                )
            )
            media_breakdown = (
                df_top.groupby(["contact_name", "mime_cat"])
                .size()
                .unstack(fill_value=0)
            )
            stats = stats.join(media_breakdown, rsuffix="_media")

        # Enforce all media columns exist to prevent N/A in UI
        for m_type in ["audio", "video", "image", "sticker"]:
            if m_type not in stats.columns:
                stats[m_type] = 0
            # Also handle if it was joined as 'audio_media' (if collision)?
            # Logic above: rsuffix='_media'.
            # If 'audio' existed in stats, it would become 'audio_media'.
            # But 'stats' has 'media', 'deleted', 'laughs'. No 'audio'.
            # So the column joined is just 'audio' (from mime_cat values).
            # Wait, unstack produces columns named 'audio', 'image', etc.
            # So stats will have 'audio', 'image'.
            # But app.py uses 'audio_media'?
            # Let me check app.py usage.
            # app.py: get_top(fun_stats, 'audio_media')
            # So app.py EXPECTS 'audio_media'.
            # My analyzer code produces 'audio' (from mime_cat values).
            # Ah! If I changed app.py earlier or if analyzer was different.
            # Let's check app.py again to be sure what column it uses.
            pass

        # Rename columns to match app.py expectation if needed
        # Or Ensure we have 'audio_media'
        # Let's map them clearly.
        rename_map = {}
        for m_type in ["audio", "video", "image", "sticker"]:
            if m_type in stats.columns:
                rename_map[m_type] = f"{m_type}_media"
            else:
                stats[f"{m_type}_media"] = 0

        if rename_map:
            stats = stats.rename(columns=rename_map)

        return stats.fillna(0)

    def get_emoji_stats(self, top_n=100, exclude_groups=False):
        """
        Returns top 5 emojis per contact.
        Warning: Can be slow on large datasets.
        """
        import emoji

        df = self.data.copy()
        if exclude_groups:
            df = df[~df["is_group"]]
        df = df[df["from_me"] == 0]  # Remove Me

        if top_n and top_n > 0:
            top_ids = df["contact_name"].value_counts().head(top_n).index
            df = df[df["contact_name"].isin(top_ids)].copy()

        # Extract emojis
        # Helper to get list of emojis
        def extract_emojis(text):
            if not isinstance(text, str):
                return []
            return [e["emoji"] for e in emoji.emoji_list(text)]

        # Filter for messages with text
        text_msgs = df[df["text_data"].notnull()]

        # Apply extraction
        # This gives a column of lists
        emoji_lists = text_msgs["text_data"].apply(extract_emojis)

        # Explode to get one row per emoji instance
        # Create temp DF
        emoji_df = pd.DataFrame(
            {"contact_name": text_msgs["contact_name"], "emoji": emoji_lists}
        ).explode("emoji")

        # Drop nulls (no emojis)
        emoji_df = emoji_df.dropna(subset=["emoji"])

        if emoji_df.empty:
            return pd.DataFrame()

        # Count per contact
        # Group by Contact + Emoji
        emoji_counts = (
            emoji_df.groupby(["contact_name", "emoji"]).size().reset_index(name="count")
        )

        # Get Top 5 per contact
        top_emojis = (
            emoji_counts.sort_values(["contact_name", "count"], ascending=[True, False])
            .groupby("contact_name")
            .head(5)
        )

        global_top = emoji_df["emoji"].value_counts().head(10)

        return {"per_contact": top_emojis, "global": global_top}

    def get_streak_stats(self, exclude_groups=False):
        """
        Calculates the longest streak of consecutive days with messages.
        Returns DataFrame: index (contact_name), streak (int), start_date, end_date
        """
        df = self.data.copy()
        if exclude_groups:
            df = df[~df["is_group"]]

        df["date"] = df["timestamp"].dt.date

        results = []

        # Group by Chat Name (Conversation)
        # Strength: This combines Me + Them into one timeline.
        chat_dates = df.groupby("chat_name")["date"].unique()

        for name, dates in chat_dates.items():
            if len(dates) < 2:
                continue

            sorted_dates = sorted(dates)
            max_streak = 1
            current_streak = 1
            best_start = sorted_dates[0]
            best_end = sorted_dates[0]

            cur_start = sorted_dates[0]

            for i in range(1, len(sorted_dates)):
                delta = (sorted_dates[i] - sorted_dates[i - 1]).days
                if delta == 1:
                    current_streak += 1
                else:
                    if current_streak > max_streak:
                        max_streak = current_streak
                        best_start = cur_start
                        best_end = sorted_dates[i - 1]

                    current_streak = 1
                    cur_start = sorted_dates[i]

            # Check last segment
            if current_streak > max_streak:
                max_streak = current_streak
                best_start = cur_start
                best_end = sorted_dates[-1]

            results.append(
                {
                    "contact_name": name,
                    "streak": max_streak,
                    "start_date": best_start,
                    "end_date": best_end,
                }
            )

        if not results:
            return pd.DataFrame()

        df_res = pd.DataFrame(results).set_index("contact_name")
        return df_res.sort_values("streak", ascending=False)

    def get_conversation_killers(self, threshold_seconds=86400, exclude_groups=False):
        """
        Who sent the last message before a long silence?
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()
        if exclude_groups:
            df = df[~df["is_group"]]

        df["time_to_next"] = df["timestamp"].shift(-1) - df["timestamp"]
        df["next_jid"] = df["jid_row_id"].shift(-1)

        valid_gap = (df["jid_row_id"] == df["next_jid"]) & (
            df["time_to_next"].dt.total_seconds() > threshold_seconds
        )

        killers = df[valid_gap]["contact_name"].value_counts()

        # Exclude Me logic:
        # If "You" is the killer...
        # If user wants to see their stats, fine. But usually Hall of Fame is others.
        if "You" in killers.index:
            killers = killers.drop("You")

        return killers

    def get_reaction_stats(self, exclude_groups=False, exclude_me=False):
        """
        Returns stats about reactions.
        (Updated to remove 'Me' from top reactors if exclude_me=True)
        """
        if "reactions_list" not in self.data.columns:
            return None

        df = self.data
        if exclude_groups:
            df = df[~df["is_group"]]

        df_reacts = (
            df[["message_row_id", "contact_name", "reactions_list", "text_data"]]
            .dropna(subset=["reactions_list"])
            .copy()
        )

        all_reactions = []
        for idx, row in df_reacts.iterrows():
            for reaction_tuple in row["reactions_list"]:
                # Handle tuple vs list format safely
                if (
                    isinstance(reaction_tuple, (list, tuple))
                    and len(reaction_tuple) >= 2
                ):
                    emoji = reaction_tuple[0]
                    sender = str(reaction_tuple[1])

                    # Robust mapping of "Self" markers to 'You'
                    if sender in ["-1", "1", "-1.0", "1.0", "None"]:
                        sender = "You"

                    # If exclude_me is True and sender is 'You', skip adding
                    if exclude_me and sender == "You":
                        continue

                    all_reactions.append(
                        {
                            "message_id": row["message_row_id"],
                            "chat_contact": row["contact_name"],
                            "reaction": emoji,
                            "sender": sender,
                            "preview": str(row["text_data"])[:50],
                        }
                    )

        if not all_reactions:
            return None

        feat_df = pd.DataFrame(all_reactions)

        # Top Reactors
        # Remove 'You' if identified?
        # Sender is raw JID or user string. "You" might be the user JID.
        # Hard to map exactly without JID table specific lookup for 'me'.
        # But usually we can assume the user JID is distinct.
        # For now, return as is. The user can visually ignore or we refine if we know 'my_jid'.
        top_reactors = feat_df["sender"].value_counts().head(10)

        top_emojis = feat_df["reaction"].value_counts().head(10)
        msg_counts = (
            feat_df.groupby(["message_id", "preview", "chat_contact"])
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="count")
        )
        return {
            "top_reactors": top_reactors,
            "top_emojis": top_emojis,
            "most_reacted": msg_counts,
        }

    def get_mention_stats(self, top_n=50, exclude_groups=False):
        """
        Analyze @mentions.
        """
        if (
            "mentions_list" not in self.data.columns
            and "mentions_pairs" not in self.data.columns
        ):
            return None

        df = self.data
        if exclude_groups:
            df = df[~df["is_group"]]

        if df.empty:
            return None

        # Preferred path: parser provides explicit (mentioned_jid_row_id, mentioned_name) tuples.
        if "mentions_pairs" in df.columns:
            df_mentions = (
                df[["contact_name", "from_me", "mentions_pairs"]]
                .dropna(subset=["mentions_pairs"])
                .copy()
            )
            if df_mentions.empty:
                return None

            df_exploded = df_mentions.explode("mentions_pairs")
            df_exploded["mentioned_jid_row_id"] = df_exploded["mentions_pairs"].apply(
                lambda p: (
                    p[0] if isinstance(p, (list, tuple)) and len(p) >= 2 else pd.NA
                )
            )
            df_exploded["mentioned_name"] = df_exploded["mentions_pairs"].apply(
                lambda p: (
                    p[1] if isinstance(p, (list, tuple)) and len(p) >= 2 else pd.NA
                )
            )
        else:
            # Fallback path: use IDs only and infer names from available sender/chat mappings.
            if "mentions_list" not in df.columns:
                return None

            df_mentions = (
                df[["contact_name", "from_me", "mentions_list"]]
                .dropna(subset=["mentions_list"])
                .copy()
            )
            if df_mentions.empty:
                return None

            df_exploded = df_mentions.explode("mentions_list")
            df_exploded["mentioned_jid_row_id"] = df_exploded["mentions_list"]

            sender_map = pd.Series(dtype="object")
            if "sender_jid_row_id" in df.columns:
                sender_map = (
                    df[["sender_jid_row_id", "contact_name"]]
                    .dropna()
                    .drop_duplicates("sender_jid_row_id")
                    .set_index("sender_jid_row_id")["contact_name"]
                )

            direct_map = pd.Series(dtype="object")
            if "jid_row_id" in df.columns and "chat_name" in df.columns:
                direct_map = (
                    df[df["is_group"] == False][["jid_row_id", "chat_name"]]
                    .dropna()
                    .drop_duplicates("jid_row_id")
                    .set_index("jid_row_id")["chat_name"]
                )

            jid_map = sender_map.combine_first(direct_map)
            df_exploded["mentioned_name"] = df_exploded["mentioned_jid_row_id"].map(
                jid_map
            )

        df_exploded = df_exploded.dropna(subset=["mentioned_name"])
        if df_exploded.empty:
            return None

        i_mention = (
            df_exploded[df_exploded["from_me"] == 1]["mentioned_name"]
            .value_counts()
            .head(top_n)
        )

        # Detect mentions of "me" by ID (preferred) and by common aliases.
        me_aliases = {"You", "Me", "Myself"}
        my_jid_ids = set()
        if "sender_jid_row_id" in df.columns:
            my_ids = pd.to_numeric(
                df.loc[df["from_me"] == 1, "sender_jid_row_id"], errors="coerce"
            ).dropna()
            my_jid_ids = {int(x) for x in my_ids if int(x) > 0}

        me_mask = df_exploded["mentioned_name"].isin(me_aliases)
        if my_jid_ids:
            mentioned_ids = pd.to_numeric(
                df_exploded["mentioned_jid_row_id"], errors="coerce"
            )
            me_mask = me_mask | mentioned_ids.isin(my_jid_ids)

        mentions_of_me = df_exploded[me_mask]["contact_name"].value_counts().head(top_n)

        return {"i_mention": i_mention, "who_mentions_me": mentions_of_me}

    def get_historical_stats(self, exclude_groups=False):
        # Returns:
        # - First Message per chat
        # - Velocity (WPM)
        df = self.data.sort_values("timestamp").copy()
        if exclude_groups:
            df = df[~df["is_group"]]

        # 1. First Message (preserve values from the same physical row).
        # groupby().first() is per-column non-null and can mix values from different rows.
        first_msgs = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["chat_name"], keep="first")
            .set_index("chat_name")[["timestamp", "text_data"]]
        )
        first_msgs = first_msgs.sort_values("timestamp")

        # 2. Velocity
        df_text = df[df["text_data"].notnull()].copy()
        df_text = df_text[df_text["from_me"] == 0]  # Velocity of OTHERS (default)

        df_text["word_count"] = df_text["text_data"].astype(str).str.split().str.len()
        df_text["minute_key"] = df_text["timestamp"].dt.floor("min")

        # Velocity per SENDER (Contact Name)
        wpm = (
            df_text.groupby(["contact_name", "minute_key"])["word_count"]
            .sum()
            .reset_index()
        )
        max_wpm = (
            wpm.groupby("contact_name")["word_count"].max().sort_values(ascending=False)
        )

        return {"first_msgs": first_msgs, "velocity_wpm": max_wpm}

    def get_true_ghosting_stats(self, threshold_hours=24, exclude_list=None):
        """
        Advanced ghosting:
        - Ghosted: I sent -> They Read it (timestamp) -> No Reply > T hours.
        - Left on Delivered: I sent -> No Read -> No Reply > T hours.
        """
        threshold_seconds = threshold_hours * 3600
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        # Robustness: Ensure read_at exists
        if "read_at" not in df.columns:
            df["read_at"] = pd.NaT

        if exclude_list:
            df = df[~df["chat_name"].isin(exclude_list)]

        df["next_from_me"] = df["from_me"].shift(-1)
        df["next_timestamp"] = df["timestamp"].shift(-1)
        df["next_jid"] = df["jid_row_id"].shift(-1)

        # Filter for MY messages
        my_msgs = df[df["from_me"] == 1].copy()

        # Refined Logic to avoid false positives at end of data
        max_ts = self.data["timestamp"].max()

        same_chat = my_msgs["jid_row_id"] == my_msgs["next_jid"]
        effective_next_ts = my_msgs["next_timestamp"].copy()
        # If switch chat or end of data, compare with Global Max Time
        effective_next_ts.loc[~same_chat] = max_ts
        effective_next_ts.loc[my_msgs["next_jid"].isna()] = max_ts

        time_diff = (effective_next_ts - my_msgs["timestamp"]).dt.total_seconds()

        is_ignored = time_diff > threshold_seconds

        ghost_candidates = my_msgs[is_ignored]

        # Classify
        # True Ghost: Read timestamp exists
        # Unseen: Read timestamp is NaT
        ghost_candidates = ghost_candidates.copy()
        ghost_candidates.loc[:, "type"] = ghost_candidates["read_at"].apply(
            lambda x: "True Ghost 👻" if pd.notnull(x) else "Left on Delivered 📨"
        )

        # Use chat_name for valid contact identification
        # True Ghosting: I sent (You) -> They (Chat Name) didn't reply.
        stats = (
            ghost_candidates.groupby(["chat_name", "type"]).size().unstack(fill_value=0)
        )
        if stats.empty:
            return pd.DataFrame()

        stats["Total Ignored"] = stats.sum(axis=1)

        # Add Gender
        them_df = self.data[self.data["from_me"] == 0]
        gender_map = them_df.drop_duplicates("chat_name").set_index("chat_name")[
            "gender"
        ]
        stats["gender"] = stats.index.map(gender_map)

        return stats.sort_values("Total Ignored", ascending=False)

    def get_advanced_reply_stats(self, limit_hours=8, reply_to=0):
        """
        Returns:
        - Distribution DataFrame (buckets)
        - Fastest/Slowest Responders (Avg Minutes)

        reply_to:
          0 = They are replying to Me (From Them, Prev From Me) -> 'Their Speed'
          1 = I am replying to Them (From Me, Prev From Them) -> 'My Speed'
        """
        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        df["prev_from_me"] = df["from_me"].shift(1)
        df["prev_jid"] = df["jid_row_id"].shift(1)
        df["delta"] = (df["timestamp"] - df["timestamp"].shift(1)).dt.total_seconds()

        # Determine strict reply condition
        # If reply_to=0 (They -> Me): Current=0, Prev=1
        # If reply_to=1 (Me -> Them): Current=1, Prev=0
        target_from = 0 if reply_to == 0 else 1
        prev_from = 1 if reply_to == 0 else 0

        # Ensure is_quote exists
        if "is_quote" not in df.columns:
            df["is_quote"] = False

        # Base Condition: Strict Alternating Sender in Same Chat
        base_condition = (
            (df["from_me"] == target_from)
            & (df["prev_from_me"] == prev_from)
            & (df["jid_row_id"] == df["prev_jid"])
        )

        # Time Condition OR Quote Condition
        # If it's a quote, we count it even if delay is huge
        valid_rows = base_condition & (
            (df["delta"] < limit_hours * 3600) | (df["is_quote"] == True)
        )

        replies = df[valid_rows].copy()

        if replies.empty:
            return None, None

        # Buckets (Detailed)
        # <1m, 1-5m, 5-15m, 15m-1h, 1h-4h, 4h-8h, >8h
        bins = [0, 60, 300, 900, 3600, 14400, 28800, max(28801, limit_hours * 3600)]
        labels = ["<1m", "1-5m", "5-15m", "15m-1h", "1h-4h", "4h-8h", ">8h"]

        # Handle small limit cases dynamically
        if limit_hours <= 1:
            bins = [0, 60, 300, 900, 3600]
            labels = ["<1m", "1-5m", "5-15m", "15m-1h"]

        try:
            replies["bucket"] = pd.cut(
                replies["delta"], bins=bins, labels=labels, duplicates="drop"
            )
        except ValueError:
            replies["bucket"] = "Unknown"

        distribution = (
            replies.groupby(["chat_name", "bucket"], observed=False)
            .size()
            .unstack(fill_value=0)
        )

        # Aggregates
        aggs = (
            replies.groupby("chat_name")["delta"].apply(
                lambda x: getattr(x, self.stat_func)()
            )
            / 60
        )  # Minutes

        # Filter sparse contacts
        counts = replies["chat_name"].value_counts()
        valid_contacts = counts[counts >= 5].index
        aggs = aggs[aggs.index.isin(valid_contacts)]

        return distribution, aggs

    def get_left_on_delivered_stats(self, threshold_seconds=86400):
        """
        Identify who YOU leave on delivered (Unread & Ignored).
        Logic: Last msg from THEM, I never read it (read_at is NaT), and silence > Threshold.
        This implies I ignored it and didn't even open the chat (or receipts off).
        """
        if "read_at" not in self.data.columns:
            return pd.DataFrame()

        df = self.data.sort_values(["jid_row_id", "timestamp"]).copy()

        df["next_from_me"] = df["from_me"].shift(-1)
        df["next_jid"] = df["jid_row_id"].shift(-1)
        df["next_timestamp"] = df["timestamp"].shift(-1)

        df["time_to_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

        their_msgs = df[df["from_me"] == 0]

        ignored = their_msgs[
            (their_msgs["next_jid"] != their_msgs["jid_row_id"])
            | (their_msgs["time_to_next"].isna())
            | (their_msgs["time_to_next"] > threshold_seconds)
        ]

        # Crucial Filter: I did NOT read it.
        # If read_at is valid, then I read it -> "Left on Read" (Ghosting).
        # We want "Left on Delivered" -> read_at is NaT.
        delivered_ignored = ignored[ignored["read_at"].isna()]

        counts = delivered_ignored["contact_name"].value_counts().head(20).reset_index()
        counts.columns = ["contact_name", "count"]

        # Add gender
        gender_map = self.data.drop_duplicates("contact_name").set_index(
            "contact_name"
        )["gender"]
        counts["gender"] = counts["contact_name"].map(gender_map)

        return counts

    def get_location_data(self):
        """
        Returns location data for mapping.
        """
        if "latitude" in self.data.columns and "longitude" in self.data.columns:
            # Filter valid locations
            locs = self.data.dropna(subset=["latitude", "longitude"]).copy()
            # Filter 0,0 if that occurs as error
            locs = locs[(locs["latitude"] != 0) | (locs["longitude"] != 0)]
            if not locs.empty:
                return locs
        return pd.DataFrame()
