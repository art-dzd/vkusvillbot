from vkusvillbot.config import Settings


def test_settings_load_env_overrides(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "telegram:\n  bot_username: vkusvillaibot\n"
        "db:\n  path: ../vkusvill.db\n"
        "llm:\n  provider: openrouter\n  model: test-model\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_PROXY_URL", "http://proxy:8888")
    monkeypatch.setenv("DB_PATH", "data/test.db")

    settings = Settings.load()
    assert settings.telegram.token == "token"
    assert settings.db.path == "data/test.db"
    assert settings.llm.api_key == "key"
    assert settings.llm.proxy_url == "http://proxy:8888"
