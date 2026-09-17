"""Finite exact-integer then RN-half reference; not native F16MMA parity."""
def index(x, n):
    if type(x) is not int or not 0 <= x < n:
        raise ValueError('index outside proven domain')
    return x


def finite_units(h):
    """Exact finite binary16 value in units of 2**-24, no float conversion.
    Nonfinites fail; signed zeros have the same mathematical value.
    """
    index(h, 65536)
    e, f = (h >> 10) & 31, h & 1023
    if e == 31:
        raise ValueError('nonfinite half outside arithmetic domain')
    n = f if e == 0 else (1024 + f) << (e - 1)
    return -n if h & 0x8000 else n


def round_half(n, scale=48):
    """Exact RN-even of n*2**-scale to finite half; overflow fails closed.
    Underflow uses gradual subnormals. Exact cancellation chooses +0.
    This is our explicit policy, not an inferred native MMA rounding mode.
    """
    if type(n) is not int or type(scale) is not int or scale < 24:
        raise ValueError('invalid fixed point')
    sign = 0x8000 if n < 0 else 0
    n = abs(n)
    if not n:
        return 0
    floor = n.bit_length() - 1 - scale
    shift = scale + max(floor - 10, -24)
    q, r = divmod(n, 1 << shift)
    if r > (1 << shift) // 2 or (shift and r == (1 << (shift - 1)) and q & 1):
        q += 1
    if floor < -14:
        return sign | q
    e = floor + 15
    if q == 2048:
        q = 1024
        e += 1
    if e >= 31:
        raise ValueError('half overflow outside arithmetic domain')
    return sign | (e << 10) | (q - 1024)


