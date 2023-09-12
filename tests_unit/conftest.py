"""PyTest Configurations for Unit-Testing

References:
- https://stackoverflow.com/questions/44377358/how-can-i-display-the-test-name-after-the-test-using-pytest
- https://docs.pytest.org/en/stable/fixture.html
"""
import pathlib

import pytest


@pytest.fixture(scope="session",)
def env_setup(monkeypatch):
    """Setup Environment Variables through PyTest's `monkeypatch` feature.

    TODO (tommy@overjet.ai): Configure this for your repository. See an example below of how to use
        the `pytest` "`monkeypatch`" feature -- postfixes.
    """
    # monkeypatch.setenv(
    #     "GCLOUD_PROJECT",
    #     "cariesdetection",
    # )
    return None
