"""Closed form of the linear gradient ramp (standard library only).

ramp_fraction(t, start, length): progress f of the ramp at time t, 0..1.
ramp_integral(t, start, length): integral of f from time 0 to t.
phase_delay(t, start, length):   integral of (1 - f) from time 0 to t.
Time 0 is the reference (the RF centre in timing.py); start may be negative.
"""


def ramp_fraction(t: float, start: float, length: float) -> float:
    if length <= 0:
        raise ValueError
    if t <= start:
        return 0.0
    if t >= start + length:
        return 1.0
    return (t - start) / length


def ramp_integral(t: float, start: float, length: float) -> float:
    if length <= 0:
        raise ValueError

    def h(x: float) -> float:
        if x <= start:
            return 0.0
        if x < start + length:
            return (x - start) ** 2 / (2 * length)
        return length / 2 + (x - (start + length))

    return h(t) - h(0.0)


def phase_delay(t: float, start: float, length: float) -> float:
    return t - ramp_integral(t, start, length)
