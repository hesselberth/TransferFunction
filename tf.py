#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 11:58:05 2026

@author: Marcel Hesselberth

Version 0.3

Transfer Function library for control system modeling and simulation.
Supports bilinear (Tustin) discretization and export to CMSIS-DSP
style biquads.
"""


import numpy as np
from numpy.polynomial import Polynomial
from math import factorial as f
from functools import cache

DEFAULTTRIM = 1e-12
PI = np.pi
PI2 = 2 * PI


@cache
def btmatrix(N, Ts=2):
    """
    Bilinear transform matrix using binomial theorem. For a transfer function
    H(s) = nump(s) / denp(s) where nump and denp are np.Polynomial's, num and
    den are their vectors of coefficients (num, den = nump.coef, denp.coef),
    the bilinear transform H(z) is given by the coefficient vectors b = num @ M
    and a = den @ M. For Ts = 2, H(z) = np.Polynomial(b) / np.Polynomial(a).
    """
    M = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            s = 0.0
            for k in range(max(j - i, 0), min(N - i, j) + 1):
                num = f(j) * f(N - j)
                den = f(k) * f(j - k) * f(N - i - k) * f(i - j + k)
                s += (num / den) * ((-1) ** k)
            M[j, i] = s * (2 / Ts) ** j
    return M

def bt(num, den, Ts):
    """
    Apply bilinear transform using precomputed matrix. np.Polynomial uses
    coefficient vectors (np.arrays) in low to high order where coef[0]
    corresponds to order 0 and [1, 2, 4] is the polynomial 1 + 2*s + 4*s**2.
    This library follows the low-to-high Polynomial convention, meaning that
    this transform will result in a polynomial in the z domain with
    low to high coefficients. According to the Z-transform theory the linear
    difference equation feedworward and feedback coefficients can simply be
    read from the Z-domain transfer function coefficients when it is
    written as a Laurent polynomial with negative powers. This means that,
    although the coefficient order of this library is low-to-high for the
    positive power polynomial coefficients, it is high-to-low for the
    b and a coefficients.
    
    This function takes coefficient arrays of the s domain numerator and
    denominator polynomials as input and returns the numerator and denominator
    coefficients of the Z-domain (positive power) polynomials, which will
    be padded to the highest order that occurs.
    
    Ts is the sample time > 0.
    """
    lnum = len(num)
    lden = len(den)
    orderplus1 = max(lnum, lden)
    order = orderplus1 - 1
    nump = np.pad(num, (0, orderplus1 - lnum))
    denp = np.pad(den, (0, orderplus1 - lden))
    M = btmatrix(order, Ts)
    B = nump @ M
    A = denp @ M
    norm = A[-1]  # normalize by the highest order denominator coefficient
    if abs(norm) < 1e-20:
        raise ValueError("Denominator became zero after bilinear transform")
    A = A / norm
    B = B / norm
    return B, A  # numerator coefficients, denominator coefficients


class TransferFunction:
    """
    Transfer Function class for control system modeling and simulation.
    Supports bilinear (Tustin) discretization and export to
    CMSIS-DSP style biquads for hardware implementation.
    """
    def __init__(self, *args, Ts=0.0, trim_tol=DEFAULTTRIM, **kwargs):
        """
        Transfer Function constructor.

        Parameters
        ----------
        *args : 0, 1 or 2 positional arguments
                - 0 args: unity gain TF (1/1)
                - 1 arg: copy another TransferFunction
                - 2 args: num, den (coefficients or Polynomial)
        Ts : float, optional (keyword)
             Sampling time ( > 0 for discrete, 0 for continuous)
             The default is 0 (continuous)
        trim_tol : float, optional (keyword)
             Tolerance for trimming small trailing coefficients
        """
        self.Ts = float(Ts)

        # Handle trim_tol from kwargs or default
        trim_tol = kwargs.get("trim_tol", trim_tol)

        if len(args) == 0:
            self.num = Polynomial([1.0])
            self.den = Polynomial([1.0])
        elif len(args) == 1:
            other = args[0]
            if not isinstance(other, TransferFunction):
                raise TypeError("Single positional argument must be a TransferFunction")
            self.num = other.num
            self.den = other.den
            self.Ts = getattr(other, 'Ts', 0.0)
        elif len(args) == 2:
            self.fromnd(args[0], args[1], trim_tol=trim_tol)
        else:
            raise ValueError("TransferFunction accepts 0, 1 (copy) or 2 \
                             positional arguments + Ts = keyword")
        self.symbol = 's' if Ts == 0 else 'z'
        self.num = Polynomial(self.num.coef, symbol = self.symbol)
        self.den = Polynomial(self.den.coef, symbol = self.symbol)

    def fromnd(self, n, d, trim_tol=DEFAULTTRIM):
        self.num = Polynomial(n) if not isinstance(n, Polynomial) else n
        self.den = Polynomial(d) if not isinstance(d, Polynomial) else d
        if len(self.num) == 0 or len(self.den) == 0:
            raise ValueError("numerator and denominator must be non-empty")
        self._trim(trim_tol)

    def _trim(self, tol):
        def trim_poly(p):
            c = p.coef.copy()
            while len(c) > 1 and abs(c[-1]) < tol:
                c = c[:-1]
            return Polynomial(c)
        self.num = trim_poly(self.num)
        self.den = trim_poly(self.den)
        
    def is_continuous(self):
        """
        Returns True if the transfer function is a continuous transfer function
        in the s domain, otherwise False.
        """
        return self.Ts == 0
    
    def is_discrete(self):
        """
        Returns True if the transfer function is a discrete transfer function
        in the z domain, otherwise False.
        """
        return self.Ts > 0

    def _acheck(self, Ts1, Ts2):
        """
        Check Ts alignment for arithmetic.
        Operations of discrete time transfer functions on continuous time
        transfer functions or other discrete time transfer functions with a
        different sampling time are not supported.
        """
        if not Ts1 == Ts2:
            if 0 in [Ts1, Ts2]:
                raise ValueError("mixed continuous / discrete")
            else:
                raise ValueError("different sampling times")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return TransferFunction(other * self.den + self.num, self.den, Ts = self.Ts)
        if isinstance(other, TransferFunction):
            self._acheck(self.Ts, other.Ts)
            return TransferFunction(self.num * other.den + other.num * self.den,
                                   self.den * other.den, Ts = self.Ts)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TransferFunction(other * self.num, self.den, Ts = self.Ts)
        if isinstance(other, TransferFunction):
            self._acheck(self.Ts, other.Ts)
            return TransferFunction(self.num * other.num, self.den * other.den, Ts = self.Ts)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TransferFunction(self.num, self.den * other, Ts = self.Ts)
        if isinstance(other, TransferFunction):
            self._acheck(self.Ts, other.Ts)
            return TransferFunction(self.num * other.den, self.den * other.num, Ts = self.Ts)
        raise NotImplementedError()

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return TransferFunction(other * self.den, self.num, Ts = self.Ts)
        raise NotImplementedError()

    def __neg__(self):
        return TransferFunction(-self.num, self.den, Ts = self.Ts)

    def __call__(self, s):
        s = np.atleast_1d(s)
        return self.num(s) / self.den(s)

    def bilinear_transform(self, fs=None, Ts=None, prewarp_freq=None):
        if fs is None and Ts is None:
            raise ValueError("Must provide either fs or Ts")
        Ts_val = Ts if Ts is not None else 1.0 / fs
        if Ts_val <= 0:
            raise ValueError("Sampling time must be > 0")

        if prewarp_freq is not None:
            w = PI2 * prewarp_freq
            Ts_val = (2 / w) * np.tan(w * Ts_val / 2)

        b, a = bt(self.num.coef, self.den.coef, Ts_val)
        return TransferFunction(b, a, Ts=Ts_val)

    def to_difference_equation(self, high_to_low=True):
        """
        Returns b (feedforward), a (feedback) coefficients.
        - high_to_low=True  → DSP standard: highest power first, a[0] ≈ 1
        - high_to_low=False → Low-to-high order
        """
        lc = self.den.coef[-1]
        if abs(lc) < 1e-20:
            raise ValueError("Denominator leading coefficient near zero")

        num_n = self.num.coef / lc
        den_n = self.den.coef / lc

        if high_to_low:
            return num_n, den_n
        else:
            return num_n[::-1], den_n[::-1]

    def export_cmsis_biquad_df2t(self, var_name="coeffs", instance_name="S", state_name="state"):
        """
        Export transfer function as CMSIS-DSP biquad cascade (DF-II Transposed).
        Improved version with robust pairing and better gain handling.
        """
        if not self.is_discrete():
            raise ValueError("export_cmsis_biquad_df2t requires discrete TF (Ts > 0)")

        # Correct DC gain (H(0))
        dc_gain = float(self(0.0))
        if abs(dc_gain) < 1e-12:
            dc_gain = 1.0  # avoid zero gain

        poles = self.poles()
        zeros = self.zeros()

        def roots_to_quadratics(roots, tol=1e-8):
            """Pair complex-conjugate roots into quadratics, leave reals as 1st-order."""
            if len(roots) == 0:
                return []
            
            # Sort by descending magnitude (standard for SOS)
            roots = sorted(roots, key=lambda r: -abs(r))
            
            quads = []
            used = set()
            
            for i, r in enumerate(roots):
                if id(r) in used:
                    continue
                    
                imag = np.imag(r)
                if abs(imag) < tol:  # real root
                    p = Polynomial.fromroots([r])
                    coef = p.coef[::-1]  # high-to-low: [a0, a1] or [b0, b1]
                    coef = np.pad(coef, (0, 3 - len(coef)), constant_values=0.0)
                    quads.append(coef)
                    used.add(id(r))
                else:
                    # Find conjugate pair
                    conj = np.conj(r)
                    found = False
                    for j in range(i + 1, len(roots)):
                        if id(roots[j]) in used:
                            continue
                        if np.abs(roots[j] - conj) < tol:
                            p = Polynomial.fromroots([r, roots[j]])
                            coef = p.coef[::-1]  # high-to-low
                            coef = np.pad(coef, (0, 3 - len(coef)), constant_values=0.0)
                            quads.append(coef)
                            used.add(id(r))
                            used.add(id(roots[j]))
                            found = True
                            break
                    if not found:
                        # Unpaired complex root (rare) → treat as real
                        p = Polynomial.fromroots([r])
                        coef = p.coef[::-1]
                        coef = np.pad(coef, (0, 3 - len(coef)), constant_values=0.0)
                        quads.append(coef)
                        used.add(id(r))
            return quads

        pole_quads = roots_to_quadratics(poles)
        zero_quads = roots_to_quadratics(zeros)

        # Balance number of sections
        n = max(len(pole_quads), len(zero_quads))
        while len(pole_quads) < n:
            pole_quads.append(np.array([1.0, 0.0, 0.0]))
        while len(zero_quads) < n:
            zero_quads.append(np.array([1.0, 0.0, 0.0]))

        # Build sections
        sections = []
        remaining_gain = dc_gain

        for i, (zq, pq) in enumerate(zip(zero_quads, pole_quads)):
            # Normalize denominator (a0 = 1)
            if abs(pq[0]) < 1e-12:
                pq = np.array([1.0, 0.0, 0.0])
            else:
                pq = pq / pq[0]

            # Apply gain: full gain on first section, 1.0 on others
            g = remaining_gain if i == 0 else 1.0
            b_sec = g * zq

            # Pad b to 3 coefficients (high-to-low)
            b_sec = np.pad(b_sec, (3 - len(b_sec), 0), constant_values=0.0)

            # CMSIS-DSP DF-II Transposed format: b0, b1, b2, -a1, -a2
            sec = [
                b_sec[0], b_sec[1], b_sec[2],
                -pq[1], -pq[2]
            ]
            sections.extend(sec)

        # Print C code
        order = len(self.den.coef) - 1
        print(f"// Generated for {self!r} (order {order})")
        print(f"// {n} biquad stage(s), DC gain = {dc_gain:.8e}")
        print(f"float32_t {var_name}[] = {{")
        print("    " + ",\n    ".join(f"{x:12.8e}f" for x in sections))
        print("};")
        print("")
        print(f"float32_t {state_name}[{2 * n}] = {{0.0f}};" )
        print(f"arm_biquad_cascade_df2T_instance_f32 {instance_name};")
        print(f"arm_biquad_cascade_df2T_init_f32(&{instance_name}, {n}, {var_name}, {state_name});")
        print("// Usage example:")
        print("// float32_t y[1];")
        print("// arm_biquad_cascade_df2T_f32(&S, &x_in, y, 1);")

    def tdfilter(self, x):
        if not hasattr(self, 'Ts') or self.Ts <= 0:
            raise ValueError("requires discrete-time transfer function")
        a = self.den.coef[::-1]
        b = self.num.coef[::-1]
        lx = len(x)
        la, lb = len(a), len(b)
        xpad = np.pad(x.astype(float), (0, lb))
        y = np.zeros(lx)
        for n in range(lx):
            for i in range(lb):
                y[n] += b[lb - 1 - i] * xpad[n - i]
            for i in range(1, la):
                y[n] -= a[la - 1 - i] * y[n - i]
        return y

    def impulse_response(self, Ts, n):
        t = np.linspace(0, (n-1)*Ts, n)
        if hasattr(self, 'Ts') and self.Ts > 0:
            x = np.zeros(n)
            x[0] = 1.0
            y = self.tdfilter(x)
            return t, y
        else:
            f = np.fft.rfftfreq(n, Ts)
            w = PI2 * f
            ha = self(1j * w)
            ir = np.fft.irfft(ha)
            return t, np.real(ir)

    def step_response(self, Ts, n):
        t = np.linspace(0, (n-1)*Ts, n)
        if hasattr(self, 'Ts') and self.Ts > 0:
            u = np.ones(n)
            y = self.tdfilter(u)
            return t, y
        else:
            tt, h = self.impulse_response(Ts, n)
            y = np.convolve(h, np.ones_like(tt), mode='full')[:n]
            return tt, y

    def delay(self, n_samples, tol=1e-12):
        if self.is_continuous():
            raise ValueError("delay requires discrete TF")
        if not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError("n_samples must be non-negative integer")
        if n_samples == 0:
            return self
        delay_den = Polynomial([0.0] * n_samples + [1.0])
        return TransferFunction(self.num, self.den * delay_den, Ts=self.Ts)

    def poles(self):
        return self.den.roots()

    def zeros(self):
        return self.num.roots()

    def pzk(self):
        return self.poles(), self.zeros(), self(0.0)

    def pzinfo(self):
        """
        Pole-zero location info for plotting / decoration.
        Returns list of (frequency_in_Hz, symbol) tuples:
          '+'  : stable real pole
          'x'  : complex conjugate pole pair
          'o'  : zero
        Frequency is always positive.
        """
        poles, zeros, _ = self.pzk()
        result = []
        processed = set()

        for p in poles:
            if id(p) in processed:
                continue
            imag = np.imag(p)
            if abs(imag) < 1e-4:
                freq = -np.real(p) / PI2
                if freq > 0:
                    result.append((freq, '+'))
            else:
                freq = abs(imag) / PI2
                result.append((freq, 'x'))
                conj = np.conj(p)
                for other in poles:
                    if np.isclose(other, conj, atol=1e-6):
                        processed.add(id(other))
                        break

        for z in zeros:
            imag = np.imag(z)
            freq = abs(np.real(z) if abs(imag) < 1e-4 else imag) / PI2
            if freq > 0:
                result.append((freq, 'o'))

        result.sort(key=lambda x: x[0])
        return result

    def __repr__(self):
        ts_str = f", Ts={self.Ts}" if self.Ts > 0 else ""
        return f"TransferFunction(num={self.num}, den={self.den}{ts_str})"

    def __str__(self):
        ns = str(self.num)
        ds = str(self.den)
        sep = "\n" + "-" * max(len(ns), len(ds)) + "\n"
        return ns + sep + ds


if __name__ == "__main__":
    H = TransferFunction([1], [0, 1])
    print(H)
    print()
    Hd = H.bilinear_transform(Ts=2)
    print(Hd)

    H = TransferFunction([1], [100, 10, 1])
    print(H)
    print()
    Hd = H.bilinear_transform(Ts=2)
    print(Hd)
    
    H = TransferFunction([1], [1, 1])
    print(H)
    print()
    Hd = H.bilinear_transform(Ts=2)
    print(Hd)
