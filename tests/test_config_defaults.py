import importlib

import bot.config as config_module


def test_live_defaults_enable_api_auth_and_rollout_gate(monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setenv("TESTNET", "False")
        scoped.delenv("API_AUTH_ENABLED", raising=False)
        scoped.delenv("ROLLOUT_ENFORCE_PRODUCTION_GATE", raising=False)

        reloaded = importlib.reload(config_module)
        assert reloaded.get_api_security_config()["auth_enabled"] is True
        assert reloaded.get_rollout_config()["enforce_production_gate"] is True

    importlib.reload(config_module)


def test_live_defaults_can_be_overridden_explicitly(monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setenv("TESTNET", "False")
        scoped.setenv("API_AUTH_ENABLED", "False")
        scoped.setenv("ROLLOUT_ENFORCE_PRODUCTION_GATE", "False")

        reloaded = importlib.reload(config_module)
        assert reloaded.get_api_security_config()["auth_enabled"] is False
        assert reloaded.get_rollout_config()["enforce_production_gate"] is False

    importlib.reload(config_module)

