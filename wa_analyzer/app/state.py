"""Typed state containers for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from wa_analyzer.src.analyzer import WhatsappAnalyzer


@dataclass(frozen=True)
class DataPaths:
    msgstore: str
    wa: str
    vcf: str

    @classmethod
    def from_base_dir(cls, base_dir: str | Path) -> "DataPaths":
        base_path = Path(base_dir)
        return cls(
            msgstore=str(base_path / "msgstore.db"),
            wa=str(base_path / "wa.db"),
            vcf=str(base_path / "contacts.vcf"),
        )


@dataclass(frozen=True)
class AnonymizationState:
    mode_label: str
    mode_key: str
    anonymize_numbers: bool
    mapping: dict[Any, Any]


@dataclass(frozen=True)
class FilterState:
    date_start: date | None
    date_end: date | None
    exclude_groups: bool
    exclude_archived: bool
    exclude_low_participation: bool
    exclude_channels: bool
    exclude_me: bool
    exclude_family_global: bool
    exclude_non_contacts: bool
    exclude_family_gender: bool
    exclude_family_behavior: bool
    family_list: tuple[str, ...]


@dataclass(frozen=True)
class SidebarState:
    paths: DataPaths
    filters: FilterState
    anonymization: AnonymizationState
    use_medians: bool
    use_longer_stats: bool
    reply_threshold_hours: int
    min_word_len: int
    exclude_emails: bool


@dataclass
class AppContext:
    raw_df: pd.DataFrame
    display_df: pd.DataFrame
    base_df: pd.DataFrame
    group_base_df: pd.DataFrame
    filtered_df: pd.DataFrame
    analyzer: WhatsappAnalyzer
    full_analyzer: WhatsappAnalyzer
    sidebar: SidebarState
    msgstore_file_signature: tuple[int, int] | None
    me_display: str

    @property
    def df_raw(self) -> pd.DataFrame:
        return self.display_df

    @property
    def df_base(self) -> pd.DataFrame:
        return self.base_df

    @property
    def df_group_base(self) -> pd.DataFrame:
        return self.group_base_df

    @property
    def msgstore_path(self) -> str:
        return self.sidebar.paths.msgstore

    @property
    def wa_path(self) -> str:
        return self.sidebar.paths.wa

    @property
    def vcf_path(self) -> str:
        return self.sidebar.paths.vcf

    @property
    def _anon_key(self) -> str:
        return self.sidebar.anonymization.mode_key

    @property
    def _anon_numbers(self) -> bool:
        return self.sidebar.anonymization.anonymize_numbers

    @property
    def exclude_groups(self) -> bool:
        return self.sidebar.filters.exclude_groups

    @property
    def exclude_me(self) -> bool:
        return self.sidebar.filters.exclude_me

    @property
    def exclude_non_contacts(self) -> bool:
        return self.sidebar.filters.exclude_non_contacts

    @property
    def exclude_family_gender(self) -> bool:
        return self.sidebar.filters.exclude_family_gender

    @property
    def exclude_family_global(self) -> bool:
        return self.sidebar.filters.exclude_family_global

    @property
    def exclude_family_behavior(self) -> bool:
        return self.sidebar.filters.exclude_family_behavior

    @property
    def family_list(self) -> tuple[str, ...]:
        return self.sidebar.filters.family_list

    @property
    def use_medians(self) -> bool:
        return self.sidebar.use_medians

    @property
    def use_longer_stats(self) -> bool:
        return self.sidebar.use_longer_stats

    @property
    def reply_threshold_hours(self) -> int:
        return self.sidebar.reply_threshold_hours

    @property
    def min_word_len(self) -> int:
        return self.sidebar.min_word_len

    @property
    def exclude_emails(self) -> bool:
        return self.sidebar.exclude_emails
