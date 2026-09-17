"""OPTIONAL explicitly named NONNATIVE feature-test policy.

Importing this module selects nothing. Caller must pass reference_nonnative()
to preprocess_firstcounter0. DIV/RCP use exact rational RN, other operations
use Python math's platform host-libm binary64 then RN F32. Neither is PTX
approximate instruction bit emulation, a native oracle, or a bounded parity
claim. Source .ftz boundaries are applied by ApproxPolicy.call.
"""
from fractions import Fraction
import math
from preprocess_candidate import ApproxPolicy, UnsupportedDomain, f32_rn, fvalue


def reference_nonnative():
    def evaluate(op, bits):
        values = tuple(fvalue(b) for b in bits)
        kind = op.split('.')[0]
        if kind in ('div', 'rcp'):
            numerator, denominator = (values[0], values[1]) if kind == 'div' else (Fraction(1), values[0])
            if denominator == 0:
                raise UnsupportedDomain('nonnative policy rejects division by zero')
            zero_sign = ((bits[0] ^ bits[1]) >> 31) if kind == 'div' else bits[0] >> 31
            return f32_rn(numerator / denominator, zero_sign)
        x = float(values[0])
        if x == 0 and bits[0] >> 31:
            x = -0.0
        function = {'lg2': math.log2, 'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos}[kind]
        try:
            result = function(x)
        except (ValueError, OverflowError) as e:
            raise UnsupportedDomain('nonnative host-libm domain') from e
        if not math.isfinite(result):
            raise UnsupportedDomain('nonnative host-libm nonfinite result')
        return f32_rn(Fraction.from_float(result), int(math.copysign(1.0, result) < 0))
    return ApproxPolicy('NONNATIVE_exact-ratio-RN_DIV-RCP__host-libm-f64-to-RN-f32_LG2-SQRT-SIN-COS', True, evaluate)
