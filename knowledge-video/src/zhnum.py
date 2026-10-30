"""Spoken Chinese numbers for narration (the TTS reads digits poorly in some contexts)."""

DIG = "零一二三四五六七八九"


def int_zh(n):
    """0..9999 -> 汉字 (e.g. 149 -> 一百四十九, 20 -> 二十, 12 -> 十二)."""
    n = int(n)
    if n < 10:
        return DIG[n]
    if n < 20:
        return "十" + (DIG[n % 10] if n % 10 else "")
    out, units = "", [(1000, "千"), (100, "百"), (10, "十")]
    zero = False
    for u, name in units:
        q = n // u % 10
        if q:
            if zero:
                out += "零"
                zero = False
            out += DIG[q] + name
        elif out:
            zero = True
    if n % 10:
        out += ("零" if zero else "") + DIG[n % 10]
    return out


def dec_zh(x, digits=2):
    """0.87 -> 零点八七."""
    s = f"{x:.{digits}f}"
    a, b = s.split(".")
    return int_zh(int(a)) + "点" + "".join(DIG[int(c)] for c in b)


def pct_zh(p, digits=0):
    """0.277 -> 百分之二十八 (digits=0) / 百分之二十七点七 (digits=1)."""
    v = p * 100
    if digits == 0:
        return "百分之" + int_zh(round(v))
    a, b = f"{v:.{digits}f}".split(".")
    return "百分之" + int_zh(int(a)) + "点" + "".join(DIG[int(c)] for c in b)
