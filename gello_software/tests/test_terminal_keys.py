import os

from gello.data_utils.terminal_keys import TerminalKeys


def _keys(lines):
    r, w = os.pipe()
    with os.fdopen(w, "w") as f:
        f.write("".join(l + "\n" for l in lines))
    return TerminalKeys(stream=os.fdopen(r))


def test_enter_s_d_map_to_start_save_discard_line_mode():
    k = _keys(["", "s", "d", "x"])
    assert k.update({"status_text": "waiting"}) == "start"
    assert k.update() == "save"
    assert k.update() == "discard"
    assert k.update() == "normal"  # unknown key ignored
    assert k.update() == "normal"  # nothing pending


def test_single_key_tokens_are_recognised():
    k = _keys([])
    assert k._is_tty is False
    # the same mapping the cbreak path uses for one-character tokens
    from gello.data_utils import terminal_keys as tk
    assert "\r" in tk._START and "\n" in tk._START
    assert "d" in tk._DISCARD and "D" in tk._DISCARD and "s" in tk._SAVE
