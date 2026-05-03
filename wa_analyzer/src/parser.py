import sqlite3
import pandas as pd
import vobject
import os


USER_TIMEZONE = "Europe/Madrid"


def _to_user_datetime(values, unit="ms"):
    """Convert WhatsApp UTC millisecond timestamps to the user's local wall time."""
    converted = pd.to_datetime(values, unit=unit, errors="coerce", utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.tz_convert(USER_TIMEZONE).dt.tz_localize(None)
    if pd.isna(converted):
        return converted
    return converted.tz_convert(USER_TIMEZONE).tz_localize(None)


class WhatsappParser:
    def __init__(self, msgstore_path, wa_path, vcf_path):
        self.msgstore_path = msgstore_path
        self.wa_path = wa_path
        self.vcf_path = vcf_path
        self.conn_msg = None
        self.conn_wa = None

    def connect(self):
        if os.path.exists(self.msgstore_path):
            self.conn_msg = sqlite3.connect(self.msgstore_path)
            # Handle encoding errors gracefully
            self.conn_msg.text_factory = lambda b: b.decode(errors="ignore")
        if os.path.exists(self.wa_path):
            self.conn_wa = sqlite3.connect(self.wa_path)
            self.conn_wa.text_factory = lambda b: b.decode(errors="ignore")

    def parse_messages(self):
        if not self.conn_msg:
            print("Message store connection not established.")
            return pd.DataFrame()

        # Query based on v3 schema analysis, joining media table and chat subject
        # Added chat.subject to identify group names
        # CHANGE: message._id as message_row_id to align with other tables
        # Try obtaining from_me from key_from_me (older/standard) or from_me (newer)
        # We will try the most common 'key_from_me' first as it is part of the PK usually.
        query_v1 = """
        SELECT 
            message._id as message_row_id,
            message.chat_row_id,
            message.key_from_me as from_me,
            message.timestamp,
            message.received_timestamp,
            message.text_data,
            message.key_id,
            chat.jid_row_id,
            chat.subject,
            message.message_type,
            message.sender_jid_row_id,
            message_media.mime_type,
            message.receipt_server_timestamp,
            qm._id as reply_to_row_id
        FROM message
        LEFT JOIN chat ON message.chat_row_id = chat._id
        LEFT JOIN message_media ON message._id = message_media.message_row_id
        LEFT JOIN message_quoted mq ON message._id = mq.message_row_id
        LEFT JOIN message qm ON mq.key_id = qm.key_id
        """

        query_v2 = """
        SELECT 
            message._id as message_row_id,
            message.chat_row_id,
            message.from_me,
            message.timestamp,
            message.received_timestamp,
            message.text_data,
            message.key_id,
            chat.jid_row_id,
            chat.subject,
            message.message_type,
            message.sender_jid_row_id,
            message_media.mime_type,
            message.receipt_server_timestamp,
            qm._id as reply_to_row_id
        FROM message
        LEFT JOIN chat ON message.chat_row_id = chat._id
        LEFT JOIN message_media ON message._id = message_media.message_row_id
        LEFT JOIN message_quoted mq ON message._id = mq.message_row_id
        LEFT JOIN message qm ON mq.key_id = qm.key_id
        """

        try:
            # Try V1 (key_from_me)
            df = pd.read_sql_query(query_v1, self.conn_msg)
        except:
            # Fallback to V2 (from_me)
            try:
                df = pd.read_sql_query(query_v2, self.conn_msg)
            except Exception as e:
                print(f"Error parsing messages: {e}")
                return pd.DataFrame()

        # Convert timestamp to datetime (typically ms in WhatsApp)
        df["timestamp"] = _to_user_datetime(df["timestamp"], unit="ms")
        return df

    def parse_jids(self):
        if not self.conn_msg:
            return pd.DataFrame()

        query = """
        SELECT _id as jid_row_id, raw_string, user FROM jid
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception:
            # Fallback for old schema
            try:
                return pd.read_sql_query(
                    "SELECT _id as jid_row_id, raw_string FROM jid", self.conn_msg
                )
            except Exception as e:
                print(f"Error parsing JIDs (Fallback): {e}")
                return pd.DataFrame()

    def parse_wa_contacts(self):
        """Parses the wa.db for display names associated with JIDs"""
        if not self.conn_wa:
            print("WA DB connection not established.")
            return pd.DataFrame()

        query = """
        SELECT jid, display_name, wa_name FROM wa_contacts
        """
        try:
            df = pd.read_sql_query(query, self.conn_wa)
            return df
        except Exception as e:
            print(f"Error parsing WA contacts: {e}")
            return pd.DataFrame()

    def parse_vcf(self):
        if not self.vcf_path or not os.path.exists(self.vcf_path):
            return {}

        contacts = {}

        # Try multiple encodings to handle VCF files from different systems
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        vcf_content = None

        for encoding in encodings_to_try:
            try:
                with open(self.vcf_path, "r", encoding=encoding, errors="ignore") as f:
                    vcf_content = f.read()
                break  # Success, stop trying
            except (UnicodeDecodeError, UnicodeError):
                continue

        if vcf_content is None:
            # Last resort: read as binary and decode with errors ignored
            try:
                with open(self.vcf_path, "rb") as f:
                    vcf_content = f.read().decode("utf-8", errors="ignore")
            except Exception:
                return {}

        # Parse vcards using vobject
        try:
            for vcard in vobject.readComponents(vcf_content):
                try:
                    name = vcard.fn.value
                    if hasattr(vcard, "tel"):
                        for tel in vcard.tel_list:
                            phone_raw = str(tel.value)
                            digits = "".join(filter(str.isdigit, phone_raw))
                            if len(digits) > 5:
                                contacts[digits] = name
                                if len(digits) > 9:
                                    contacts[digits[-9:]] = name
                except Exception:
                    pass
        except Exception:
            pass

        # IMPORTANT: Manual Fallback to catch 'MOBILE' or unlabeled numbers that vobject might skip or malform
        # Iterate line by line
        lines = vcf_content.splitlines()
        current_name = None

        for line in lines:
            line = line.strip()
            if line.startswith("FN:"):
                current_name = line[3:].strip()
            elif line.startswith("N:") and not current_name:
                # Fallback name if FN missing (N:Last;First;...)
                parts = line[2:].strip().split(";")
                current_name = " ".join(
                    [p for p in parts if p][::-1]
                ).strip()  # Simple First Last

            elif line.startswith("TEL"):
                # Extract number
                # Format: TEL;TYPE=CELL:+123456 or TEL:+12345
                parts = line.split(":")
                if len(parts) > 1 and current_name:
                    phone_raw = parts[1]
                    digits = "".join(filter(str.isdigit, phone_raw))
                    if len(digits) > 5:
                        contacts[digits] = current_name
                        if len(digits) > 9:
                            contacts[digits[-9:]] = current_name

        return contacts

    def parse_reactions(self):
        """
        Parses reactions via message_add_on tables.
        Returns DataFrame with: message_row_id, reaction, sender_jid_row_id, from_me, reaction_timestamp
        """
        if not self.conn_msg:
            return pd.DataFrame()

        # message_add_on_reaction -> message_add_on -> message
        # IMPORTANT: Include 'from_me' to handle 'You' correctly
        # message_add_on_reaction -> message_add_on -> message -> chat
        # We need chat.jid_row_id to identify sender in 1-1 chats (where sender_jid_row_id is often NULL)
        query = """
        SELECT 
            mao.parent_message_row_id as message_row_id,
            mar.reaction,
            mao.sender_jid_row_id,
            mao.from_me,
            mao.timestamp as reaction_timestamp,
            chat.jid_row_id as chat_jid_row_id
        FROM message_add_on_reaction mar
        JOIN message_add_on mao ON mar.message_add_on_row_id = mao._id
        LEFT JOIN message m ON mao.parent_message_row_id = m._id
        LEFT JOIN chat ON m.chat_row_id = chat._id
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception as e:
            print(f"Error parsing reactions: {e}")
            return pd.DataFrame()

    def parse_receipts(self):
        """
        Parses reading receipts.
        Returns DataFrame with: message_row_id, receipt_mode, read_timestamp, played_timestamp
        """
        if not self.conn_msg:
            return pd.DataFrame()

        # Schema for receipts: _id, key_remote_jid, key_id, ...
        # Join with message table to link key_id -> message_row_id
        query = """
        SELECT 
            m._id as message_row_id,
            r.read_device_timestamp as read_timestamp,
            r.played_device_timestamp as played_timestamp
        FROM receipts r
        JOIN message m ON r.key_id = m.key_id
        WHERE r.read_device_timestamp > 0 OR r.played_device_timestamp > 0
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            if df.empty:
                return df

            # Normalize to numeric first.
            df["read_timestamp"] = pd.to_numeric(df["read_timestamp"], errors="coerce")
            df["played_timestamp"] = pd.to_numeric(
                df["played_timestamp"], errors="coerce"
            )

            # Keep only meaningful positive values.
            df["read_timestamp"] = df["read_timestamp"].where(df["read_timestamp"] > 0)
            df["played_timestamp"] = df["played_timestamp"].where(
                df["played_timestamp"] > 0
            )

            # Use read timestamp primarily, fallback to played timestamp when needed.
            # Then take the earliest event per message (first meaningful read/played).
            df["effective_read_timestamp"] = df["read_timestamp"].fillna(
                df["played_timestamp"]
            )
            df = df[df["effective_read_timestamp"].notna()]

            if df.empty:
                return pd.DataFrame(columns=["message_row_id", "read_timestamp"])

            first_receipt = (
                df.groupby("message_row_id", as_index=False)["effective_read_timestamp"]
                .min()
                .rename(columns={"effective_read_timestamp": "read_timestamp"})
            )
            return first_receipt
        except Exception as e:
            print(f"Error parsing receipts: {e}")
            return pd.DataFrame()

    def parse_mentions(self):
        if not self.conn_msg:
            return pd.DataFrame()
        query = """
        SELECT message_row_id, jid_row_id as mentioned_jid_row_id FROM message_mentions
        """
        try:
            return pd.read_sql_query(query, self.conn_msg)
        except:
            return pd.DataFrame()

    def parse_locations(self):
        if not self.conn_msg:
            return pd.DataFrame()
        query = """
        SELECT message_row_id, latitude, longitude, place_name, place_address FROM message_location
        """
        try:
            return pd.read_sql_query(query, self.conn_msg)
        except:
            return pd.DataFrame()

    def get_merged_data(self):
        self.connect()

        # 1. Get Messages
        messages_df = self.parse_messages()
        if messages_df.empty:
            return pd.DataFrame()

        # 2. Get Receipts
        receipts_df = self.parse_receipts()

        # 3. Merge Receipts
        # receipts_df has message_row_id (from _id or key_id join).
        # But wait, parse_receipts (Step 1610) selects m._id as message_row_id.
        if not receipts_df.empty:
            # parse_receipts already returns one row per message using the earliest
            # meaningful read/played timestamp.
            messages_df = pd.merge(
                messages_df,
                receipts_df[["message_row_id", "read_timestamp"]],
                left_on="message_row_id",
                right_on="message_row_id",
                how="left",
            )
            messages_df.rename(columns={"read_timestamp": "read_at"}, inplace=True)
        else:
            messages_df["read_at"] = pd.NaT

        # Normalize receipt-derived read_at (numeric unix ms) before fallback fills.
        if (
            "read_at" in messages_df.columns
            and not pd.api.types.is_datetime64_any_dtype(messages_df["read_at"])
        ):
            read_at_numeric = pd.to_numeric(messages_df["read_at"], errors="coerce")
            mask_invalid = read_at_numeric.isna() | (read_at_numeric <= 0)
            messages_df["read_at"] = _to_user_datetime(
                read_at_numeric.where(~mask_invalid), unit="ms"
            )

        # Fallback: Use receipt_server_timestamp if read_at is missing
        if "receipt_server_timestamp" in messages_df.columns:
            # Ensure types match
            if not pd.api.types.is_datetime64_any_dtype(
                messages_df["receipt_server_timestamp"]
            ):
                mask_invalid = messages_df["receipt_server_timestamp"].isna() | (
                    messages_df["receipt_server_timestamp"] <= 0
                )
                messages_df["receipt_server_timestamp"] = _to_user_datetime(
                    messages_df["receipt_server_timestamp"].where(~mask_invalid),
                    unit="ms",
                )

            messages_df["read_at"] = messages_df["read_at"].fillna(
                messages_df["receipt_server_timestamp"]
            )

        # Fallback for incoming messages: use received_timestamp when read_at is missing
        if "received_timestamp" in messages_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(
                messages_df["received_timestamp"]
            ):
                mask_invalid = messages_df["received_timestamp"].isna() | (
                    messages_df["received_timestamp"] <= 0
                )
                messages_df["received_timestamp"] = _to_user_datetime(
                    messages_df["received_timestamp"].where(~mask_invalid),
                    unit="ms",
                )
            incoming_mask = (messages_df["from_me"] == 0) & (
                messages_df["read_at"].isna()
            )
            messages_df.loc[incoming_mask, "read_at"] = messages_df.loc[
                incoming_mask, "received_timestamp"
            ]

        # 2. Get JIDs (to link chat_row_id -> raw_string)
        jids_df = self.parse_jids()

        # 3. Merge JIDs onto messages
        # Chat JID (Group or 1-on-1 counterpart)
        merged = pd.merge(messages_df, jids_df, on="jid_row_id", how="left")

        # 3b. Merge JIDs for SENDER (for Groups)
        # Rename columns to avoid collision
        senders = jids_df.rename(
            columns={
                "jid_row_id": "sender_jid_row_id",
                "raw_string": "sender_string",
                "user": "sender_user",
            }
        )
        merged = pd.merge(merged, senders, on="sender_jid_row_id", how="left")

        # 4. Get Names from WA DB
        wa_contacts_df = self.parse_wa_contacts()

        # 5. Get Names from VCF (Dictionary)
        vcf_contacts = self.parse_vcf()

        # 6. Resolve Names
        # Optimization: Create a JID -> Name map first for O(1) lookup
        # This avoids applying the function row-by-row for every message if we can map JIDs.

        # 6a. Helper Name Resolver
        def get_name_from_id(jid_raw, user_raw, subject=None, is_group=False):
            # 1. Group Subject
            if subject and isinstance(subject, str):
                return subject

            # 2. Group JID fallback
            if isinstance(jid_raw, str) and jid_raw.endswith("@g.us"):
                return "Unknown Group"

            # 3. Valid JID check
            if not isinstance(jid_raw, str):
                return "Unknown"

            # 4. Clean JID
            phone_part = jid_raw.split("@")[0]

            # 5. VCF (Fastest)
            if phone_part in vcf_contacts:
                return vcf_contacts[phone_part]

            # 6. Fuzzy VCF
            if len(phone_part) > 9:
                suffix = phone_part[-9:]
                if suffix in vcf_contacts:
                    return vcf_contacts[suffix]

            # 7. WA DB
            if not wa_contacts_df.empty:
                match = wa_contacts_df[wa_contacts_df["jid"] == jid_raw]
                if not match.empty:
                    disp = match.iloc[0]["display_name"]
                    if disp:
                        return disp

            return user_raw or phone_part

        # 6b. Apply to CHAT (The Conversation)
        # For Chat Name, we use 'raw_string' (the Chat JID) and 'subject'.
        merged["chat_name"] = merged.apply(
            lambda x: get_name_from_id(
                x.get("raw_string"), x.get("user"), x.get("subject")
            ),
            axis=1,
        )

        # 6c. Apply to SENDER (The Person)
        def resolve_sender(row):
            if row.get("from_me") == 1:
                return "You"

            # Use Sender JID if available (Groups), else Chat JID (1-on-1)
            target_jid = row.get("sender_string")
            target_user = row.get("sender_user")

            if not isinstance(target_jid, str):
                target_jid = row.get("raw_string")
                target_user = row.get("user")

            # Pass None for subject because Sender Name shouldn't be "Group Name"
            return get_name_from_id(target_jid, target_user, subject=None)

        merged["contact_name"] = merged.apply(resolve_sender, axis=1)

        # 7. Final Cleanup
        # Ensure timestamp
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])

        # --- Advanced Data Merging ---

        # Note: Receipts already merged in Step 3 above (lines 244-276).
        # Ensure read_at column exists and is properly typed.
        if "read_at" not in merged.columns:
            merged["read_at"] = pd.NaT
        else:
            merged["read_at"] = pd.to_datetime(merged["read_at"], errors="coerce")

        # 8. Merge Locations
        locs = self.parse_locations()
        if not locs.empty:
            merged = pd.merge(merged, locs, on="message_row_id", how="left")

        # 9. Merge Reactions (Aggregation: List of reactions string?)
        # A message can have multiple reactions. We'll store them as a list or string.
        reactions = self.parse_reactions()
        if not reactions.empty:
            # Group to list of reactions per message
            # Also want who reacted? For now, just the reactions themselves for "Most Reacted Msg"
            # Or if we want "Top Reactor", we need the sender logic.
            # Since 'merged' is per-message, merging 'list of reactions' is best.
            # But calculating "Top Reactor" requires the granular dataframe.
            # We can return the raw reactions DF separately?
            # OR we can attach a summary to the main DF and logic inside Analyzer can query raw if needed?
            # Current architecture: Parser returns ONE dataframe.
            # Let's attach 'reactions_data' as a JSON-like object or list of tuples? (sender, reaction)

            # Let's Resolve Reaction Sender Name
            # We need to map reaction_sender_jid -> Name

            # --- FALLBACK FOR 1-1 CHATS ---
            # In 1-1 chats, 'sender_jid_row_id' is often NULL for incoming reactions (implied to be chat partner).
            if "chat_jid_row_id" in reactions.columns:
                # If sender is missing (NaN or <=0) AND it's NOT from me (incoming)
                # Then the sender IS the chat counterpart (chat_jid_row_id)
                mask_missing = (reactions["sender_jid_row_id"].fillna(0) <= 0) & (
                    reactions["from_me"] == 0
                )
                reactions.loc[mask_missing, "sender_jid_row_id"] = reactions.loc[
                    mask_missing, "chat_jid_row_id"
                ]

            # Now merge to get JID string
            reactions = pd.merge(
                reactions,
                jids_df,
                left_on="sender_jid_row_id",
                right_on="jid_row_id",
                how="left",
            )

            # Create a robust sender column
            # Logic: If from_me=1, sender is 'You'.
            # Resolve names (Full Contact Lookup)
            reactions["sender_display"] = reactions.apply(
                lambda x: (
                    "You"
                    if x.get("from_me") == 1
                    else get_name_from_id(x.get("raw_string"), x.get("user"), None)
                ),
                axis=1,
            )

            # FALLBACK: If sender is explicitly '-1' (id) or string '-1', it's likely 'You' (self)
            reactions["sender_display"] = reactions["sender_display"].apply(
                lambda x: "You" if str(x) == "-1" or str(x) == "1" else x
            )

            reactions_agg = (
                reactions.groupby("message_row_id")
                .apply(
                    lambda x: list(zip(x["reaction"], x["sender_display"])),
                    include_groups=False,
                )
                .reset_index(name="reactions_list")
            )

            merged = pd.merge(merged, reactions_agg, on="message_row_id", how="left")

        # 10. Merge Mentions
        mentions = self.parse_mentions()
        if not mentions.empty:
            # Resolve mentioned JID IDs to names using the same contact resolver.
            mention_jids = jids_df.rename(
                columns={
                    "jid_row_id": "mentioned_jid_row_id",
                    "raw_string": "mentioned_raw_string",
                    "user": "mentioned_user",
                }
            )
            mentions = pd.merge(
                mentions, mention_jids, on="mentioned_jid_row_id", how="left"
            )
            mentions["mentioned_name"] = mentions.apply(
                lambda x: get_name_from_id(
                    x.get("mentioned_raw_string"), x.get("mentioned_user"), subject=None
                ),
                axis=1,
            )

            mentions_ids = (
                mentions.groupby("message_row_id")["mentioned_jid_row_id"]
                .apply(list)
                .reset_index(name="mentions_list")
            )
            mentions_names = (
                mentions.groupby("message_row_id")["mentioned_name"]
                .apply(list)
                .reset_index(name="mentions_name_list")
            )
            mentions_pairs = (
                mentions.groupby("message_row_id")
                .apply(
                    lambda x: list(zip(x["mentioned_jid_row_id"], x["mentioned_name"])),
                    include_groups=False,
                )
                .reset_index(name="mentions_pairs")
            )

            mentions_agg = mentions_ids.merge(
                mentions_names, on="message_row_id", how="left"
            )
            mentions_agg = mentions_agg.merge(
                mentions_pairs, on="message_row_id", how="left"
            )
            merged = pd.merge(merged, mentions_agg, on="message_row_id", how="left")

        # 11. Merge Quotes (to identify "Deep Replies")
        quotes = self.parse_quotes()
        if not quotes.empty:
            # We just need to know IF it is a quote
            quotes["is_quote"] = True
            merged = pd.merge(
                merged,
                quotes[["message_row_id", "is_quote"]],
                on="message_row_id",
                how="left",
            )
            merged["is_quote"] = merged["is_quote"].eq(True)
        else:
            merged["is_quote"] = False

        return merged

    def parse_quotes(self):
        """
        Parses message_quoted table to identify messages that quote others.
        """
        if not self.conn_msg:
            return pd.DataFrame()
        query = "SELECT message_row_id FROM message_quoted"
        try:
            return pd.read_sql_query(query, self.conn_msg)
        except:
            return pd.DataFrame()
