from salvador_transcriber.config import Settings


def test_openai_model_has_safe_default_and_preserves_explicit_override() -> None:
    assert Settings(_env_file=None).openai_model == "gpt-4o-transcribe"

    overridden = Settings(_env_file=None, openai_model="gpt-live-transcribe")

    assert overridden.openai_model == "gpt-live-transcribe"
