import os


def clear_console() -> None:
    """Clears the terminal screen for a clean UI."""
    os.system("cls" if os.name == "nt" else "clear")
