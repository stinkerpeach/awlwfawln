from machine import Pin, I2C
import utime
import sys
import select

BPM      = 120
STEP_MS  = int((60000 / BPM) / 4)
PRESS_MS = 60
DAC_VCC  = 2.5

def make_pin(n):
    return Pin(n, Pin.IN, pull=None)

SOUND_PINS = {
    1:  make_pin(0),
    2:  make_pin(1),
    3:  make_pin(2),
    4:  make_pin(3),
    5:  make_pin(4),
    6:  make_pin(5),
    7:  make_pin(6),
    8:  make_pin(7),
    9:  make_pin(8),
    10: make_pin(16),
    11: make_pin(17),
    12: make_pin(18),
}

i2c          = I2C(0, sda=Pin(12), scl=Pin(13), freq=400000)
MCP4725_ADDR = 0x62

def set_dac(voltage):
    voltage = max(0.0, min(float(voltage), DAC_VCC))
    value   = int((voltage / DAC_VCC) * 4095)
    i2c.writeto(MCP4725_ADDR, bytes([
        0x40,
        (value >> 4) & 0xFF,
        (value << 4) & 0xFF,
    ]))

PATTERNS = {
    1:  ["1000000000000000","1000000010000000","1000000000001000","1000100010001000","1001000110010000","1001001110011000"],  # kick
    2:  ["0000000010000000","0000000010001000","0000100010100000","0000100100100010","0010100100001001","0010100100101001"],  # snare
    3:  ["1000100010001000","1010101010101010","1100110010011100","1011101010111010","1011101110111011","1111101111110111"],  # closed hat
    4:  ["1000000000000000","0000000101000000","0000000101000001","0000101000001000","0001001000010100","0100100101001001"],  # open hat
    5:  ["1000000000000000","0000000010000000","0000000010000001","0010000000100000","0010010000100100","0010110000101100"],  # sticks
    6:  ["0000000010000000","0000100000000000","0000100000001000","0000100100001000","0000100100001001","0010100101001001"],  # clap
    7:  ["1000000000000000","0000100000001000","1000100010001000","1010101010101010","1011101010111010","1111111111111111"],  # click
    8:  ["1000000000000000","0000000000000001","0000000100000001","0000000100010000","0000010000000100","0010000001000010"],  # low tom
    9:  ["1000000000000000","0001000000000000","0001000000010000","1000000000010000","1000100000010000","1000100010001000"],  # hi tom
    10: ["1000000000000000","1000000000001000","1000000000000010","1000100010001000","1000100110001001","1010100110101001"],  # cowbell
    11: ["1000000000000000","0010000000000000","0000001000000010","0000001000100000","0000100000001000","0000010001000100"],  # tone
    12: ["1000000000000000","1010000000000000","1000000010000010","1010101010101010","1010111010101110","1110101011101110"],  # bass
}

SILENT = "0000000000000000"

rock_complexity = {rid: 0 for rid in PATTERNS}

def press(pin, ms=PRESS_MS):
    pin.init(Pin.OUT, value=0)
    utime.sleep_ms(ms)
    pin.init(Pin.IN, pull=None)

def tick(step):
    active = []
    for rid, pin in SOUND_PINS.items():
        cx = rock_complexity[rid]
        if cx == 0:
            continue   # rock removed — silent
        pattern = PATTERNS[rid][cx - 1]
        if pattern[step] == "1":
            active.append(pin)

    if not active:
        return

    for pin in active:
        pin.init(Pin.OUT, value=0)
    utime.sleep_ms(PRESS_MS)
    for pin in active:
        pin.init(Pin.IN, pull=None)

def handle_message(line):
    line = line.strip()
    if not line:
        return
    try:
        parts   = line.split(",")
        rid     = int(parts[0])
        cx      = int(parts[1])      # 0 = silent, 1-6 = active
        voltage = float(parts[2])
        if rid in rock_complexity and 0 <= cx <= 6:
            rock_complexity[rid] = cx
            if cx > 0:
                set_dac(voltage)
            print(f"ok rock{rid} cx={cx} v={voltage:.3f}")
        else:
            print(f"err: {line}")
    except Exception as e:
        print(f"err: {e}")

# main loop

print(f"Ready. BPM={BPM} STEP_MS={STEP_MS}ms")

step = 0

while True:
    t_start  = utime.ticks_ms()
    tick(step)
    step     = (step + 1) % 16
    deadline = utime.ticks_add(t_start, STEP_MS)
    while utime.ticks_diff(deadline, utime.ticks_ms()) > 2:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line:
                handle_message(line)
        utime.sleep_ms(1)