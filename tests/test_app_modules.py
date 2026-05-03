import importlib


def test_modular_app_modules_import():
    modules = [
        "wa_analyzer.app.main",
        "wa_analyzer.app.sidebar",
        "wa_analyzer.app.privacy",
        "wa_analyzer.app.db_loaders",
        "wa_analyzer.app.filters",
        "wa_analyzer.app.race_video",
        "wa_analyzer.app.state",
        "wa_analyzer.app.tabs.activity",
        "wa_analyzer.app.tabs.behavioral",
        "wa_analyzer.app.tabs.gender",
        "wa_analyzer.app.tabs.wordcloud",
        "wa_analyzer.app.tabs.chat_explorer",
        "wa_analyzer.app.tabs.group_explorer",
        "wa_analyzer.app.tabs.fun_insights",
        "wa_analyzer.app.tabs.map_view",
        "wa_analyzer.app.tabs.chat_viewer",
        "wa_analyzer.app.tabs.inbox_triage",
    ]
    for module in modules:
        assert importlib.import_module(module)


def test_app_entrypoint_imports_in_bare_mode():
    assert importlib.import_module("app")

