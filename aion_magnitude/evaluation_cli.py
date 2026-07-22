"""Module entry point for ``python -m aion_magnitude.evaluation_cli``."""

from .evaluation.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
