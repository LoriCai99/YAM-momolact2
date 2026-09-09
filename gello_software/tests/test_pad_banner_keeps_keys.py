"""A key pressed while a pad banner is showing must not be lost."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
import pytest

from gello.data_utils import keyboard_interface as ki


@pytest.fixture
def kb():
    try:
        k = ki.KBReset()
    except pygame.error as e:  # no display at all
        pytest.skip(str(e))
    yield k
    pygame.quit()


def test_key_during_banner_survives(kb):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0, unicode="d", scancode=0))
    kb.banner("CAMERA STALE", None, duration_s=0.1)   # drains the queue
    assert kb.update() == "discard"                     # ...but the key still counts
