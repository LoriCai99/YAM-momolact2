"""channel: may be an adapter USB serial; canN names swap on every re-plug."""
import os

import pytest

from gello.robots.yam import resolve_can_channel


def test_literal_interface_passes_through():
    assert resolve_can_channel("can0") == "can0"
    assert resolve_can_channel("can7") == "can7"


def test_serial_resolves_to_the_interface_that_has_it(tmp_path, monkeypatch):
    # Fake /sys/class/net/canN/device -> .../<usb dev>/serial
    for name, serial in (("can0", "AAA"), ("can1", "BBB")):
        usb = tmp_path / f"usb-{name}"
        (usb / "child").mkdir(parents=True)
        (usb / "serial").write_text(serial + "\n")
        net = tmp_path / "net" / name
        net.mkdir(parents=True)
        os.symlink(usb / "child", net / "device")
    monkeypatch.setattr("glob.glob", lambda pat: [str(tmp_path / "net" / "can0"), str(tmp_path / "net" / "can1")])
    assert resolve_can_channel("BBB") == "can1"
    assert resolve_can_channel("serial:AAA") == "can0"
    with pytest.raises(RuntimeError, match="no CAN adapter"):
        resolve_can_channel("serial:MISSING")
