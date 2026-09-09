import io

from gello.data_utils.terminal_keys import TerminalKeys


class _Stdin(io.StringIO):
    def fileno(self):  # select() needs a real fd; use a pipe instead
        raise ValueError


def _keys(lines):
    import os
    r, w = os.pipe()
    with os.fdopen(w, "w") as f:
        f.write("".join(l + "\n" for l in lines))
    return TerminalKeys(stream=os.fdopen(r))


def test_enter_s_d_map_to_start_save_discard():
    k = _keys(["", "s", "d", "x"])
    assert k.update({"status_text": "waiting"}) == "start"
    assert k.update() == "save"
    assert k.update() == "discard"
    assert k.update() == "normal"  # unknown key ignored
    assert k.update() == "normal"  # nothing pending
