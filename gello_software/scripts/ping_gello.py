"""Broadcast-ping the GELLO Dynamixel chains on every FTDI port.

Diagnostic for "teleop hangs at startup". The GELLO FT232H adapter is USB-bus
powered, so /dev/ttyUSB* appearing proves nothing about the servo bus -- the
Dynamixels need their own supply. This pings every FTDI port at the common
baudrates and prints which servo IDs answer.

Expect ids=[1,2,3,4,5,6,7] on the left port and [8,9,10,11,12,13,14] on the
right. Empty lists on every row = the Dynamixel power rail is off or the servo
cable is not seated.

Unlike DynamixelDriver this never blocks: it uses broadcastPing directly and
never enters the unbounded `while self._joint_angles is None` spin in
gello/dynamixel/driver.py.

Usage:  python scripts/ping_gello.py
"""

import glob

from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler

BAUDRATES = [57600, 1000000, 2000000, 115200, 3000000, 4000000]
GELLO_GLOB = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_*"


def main() -> None:
    ports = sorted(glob.glob(GELLO_GLOB))
    if not ports:
        print(f"No FTDI adapters found matching {GELLO_GLOB}")
        print("The GELLO leader USB is not connected (or ftdi_sio did not bind).")
        return

    print(f"Found {len(ports)} FTDI adapter(s):")
    for p in ports:
        print(f"  {p}")
    print()

    found_any = False
    for port in ports:
        label = port.split("Converter_")[-1]
        for baud in BAUDRATES:
            ph = PortHandler(port)
            pk = PacketHandler(2.0)
            if not ph.openPort():
                print(f"{label} @{baud:>7}: cannot open port")
                continue
            if not ph.setBaudRate(baud):
                print(f"{label} @{baud:>7}: cannot set baudrate")
                ph.closePort()
                continue
            ids, _ = pk.broadcastPing(ph)
            ph.closePort()
            ids = sorted(ids) if ids else []
            if ids:
                found_any = True
            print(f"{label} @{baud:>7}: ids={ids}")
        print()

    if not found_any:
        print("No Dynamixel servos answered on any port at any baudrate.")
        print("=> Check the GELLO servo power supply and that the 3-pin servo")
        print("   cable is seated in the adapter. USB enumeration alone does")
        print("   NOT mean the servo bus is powered.")


if __name__ == "__main__":
    main()
