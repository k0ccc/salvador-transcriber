from importlib.resources import files


def demo_html() -> str:
    """Load the packaged static page without depending on the process cwd."""

    return (
        files("salvador_transcriber.demo").joinpath("static/index.html").read_text(encoding="utf-8")
    )
