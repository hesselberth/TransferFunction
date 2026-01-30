#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:31:33 2025

@author: Marcel Hesselberth
"""

import numpy as np
from numpy.polynomial import Polynomial
from info import Info
from functools import cache
from math import factorial as f

DEFAULTTRIM = 1e-12
info = Info()
PI = np.pi
PI2 = 2 * PI

@cache
def btmatrix(N, Ts=2):
    """
    Generate a matrix for the bilinear transform of general order.
    Uses the binomial theorem to compute the coefficients.

    Parameters
    ----------
    N  : Integer
         Order N is the highest polynomial order in the transfer function.
    Ts : Float, optional
         Sampling time. The default is 2, corresponding to K = 2 / 2 = 1.

    Returns
    -------
    M : The bilinear transform matrix M
        For a vector A of polynomial coefficients A @ M is 
        the bilinear transform. N is the order of the polynomial.
        len(A) == N+1.
        The coefficients in M are arranged such that A[0] is order 0.
    """
    M = np.zeros([N+1,N+1])
    for i in range(N+1):
        for j in range(N+1):
            acc = 0;
            for k in range(max(i+j-N, 0), min(i, j)+1):
                acc += ( ( f(j)*f(N-j) ) /
                        ( f(k)*f(j-k)*f(i-k)*f(N-j-i+k) ) ) * pow(-1, k)
            M[j][N-i] = acc *pow(2/Ts, j)
    return M


def bt(num, den, Ts):
    """
    Bilinear transform of a transfer function for sample time Ts.

    Parameters
    ----------
    num : np.array of polynomial coefficients
          The numerator polynomial of the transfer function.
    den : np.array
          The denominator polynomial of the transfer function.
    Ts  : Float  
          The sample interval.

    Returns
    -------
    A : np.array
        The coefficients of the numerator polynomial.
    B : np.array
        The coefficients of the denominator polynomial.
        
    A and B both have length max(len(num), len(den))  
    """
    la = len(den)
    lb = len(num)
    orderplus1 = max(la, lb)
    order = orderplus1 - 1
    a = np.pad(den, (0, orderplus1 - la))
    b = np.pad(num, (0, orderplus1 - lb))
    M = btmatrix(order, Ts)
    A = a @ M
    B = b @ M / A[0]
    A /= A[0]
    return B, A


# Generic transfer function class.
class TransferFunction:
    def __init__(self, *args, **kwargs):
        """
        Transfer Function constructor.

        Parameters
        ----------
        *args : Pass either nothing for a unity transfer function, a tuple
                (num, den) containing the numerator and denominator,
                num, den as a numerator and denominator,
                num, den, Ts. Ts is a sampling time of a discrete time system.
                num and den may be sequences of coefficients of polynomials
                (0th order first) or np.Polynomial's.
        **kwargs : -
                not currently used.

        Returns
        -------
        None.

        """
        if "trim_tol" in kwargs:
            trim_tol = kwargs["trim_tol"]
        else:
            trim_tol = DEFAULTTRIM
        if args == ():
            self.num = Polynomial([1])
            self.den = Polynomial([1])
        elif len(args) == 1:
            tf = args[0]
            num = tf.num
            den = tf.den
            self.fromnd(num, den)
        elif len(args) in [2,3]:
            self.fromnd(args[0], args[1])
            if len(args) == 3:
                self.Ts = args[2]
            else:
                self.Ts = 0
        else:
            info.error("Transferfunction constructor requires 0 or 2 arguments.")
        self.a = self.b = None

    def fromnd(self, n, d, trim_tol=DEFAULTTRIM):
        """
        Helper to initialize a TransferFunction.
        Called by the constructor.

        Parameters
        ----------
        n : sequence or np.Polynomial
            numerator
        d : sequence or np.Polynomial
            denominator
        trim_tol : Float, optional
            Ignore small coefficients (due to prior numerical error).
            The default is DEFAULTTRIM.

        Returns
        -------
        None.

        """
        self.num = Polynomial(n) if not isinstance(n, Polynomial) else n
        self.den = Polynomial(d) if not isinstance(d, Polynomial) else d
        if len(self.num) < 1 or len(self.den) < 1:
            info.error("b and a must be non-empty coefficient lists")
        self._trim(trim_tol)

    def _trim(self, tol=1e-12):
        """
        Remove negligibe coefficients.

        Parameters
        ----------
        tol : Float, optional
              Numerical tolerance. The default is 1e-12.

        Returns
        -------
        None.

        """
        def trim_poly(p):
            coeffs = p.coef.copy()
            while len(coeffs) > 1 and np.abs(coeffs[-1]) < tol:
                coeffs = coeffs[:-1]
            return Polynomial(coeffs)
        self.num = trim_poly(self.num)
        self.den = trim_poly(self.den)

    def __add__(self, other):
        """
        Add 2 transfer functions.

        Parameters
        ----------
        other : TransferFunction or a number
                The transfer function to be added.

        Returns
        -------
        TransferFunction
                The sum.
        """
        if type(other) in [int, float]:
            return TransferFunction(other * 
                self.den.coef[0] + self.num, self.den)
        if isinstance(other, TransferFunction):
            return TransferFunction(self.num*other.den + 
                other.num*self.den, self.den*other.den)

    def __radd__(self, other): return self + other
    
    def __sub__(self, other): return self + (-other)
    
    def __mul__(self, other):
        """
        Multiply 2 transfer functions.

        Parameters
        ----------
        other : TransferFunction or a number
                The multiplier.

        Returns
        -------
        TransferFunction
                The product.
        """
        if type(other) in [int, float]:
            return TransferFunction(other * self.num, self.den)
        if isinstance(other, TransferFunction):
            return TransferFunction(self.num*other.num, self.den*other.den)

    def __rmul__(self, other): return self * other
    
    def __truediv__(self, other):
        """
        Divide a transfer function.

        Parameters
        ----------
        other : TransferFunction or a number 
                The denominator.

        Returns
        -------
        TransferFunction
                The quotient.
        """
        if type(other) in [int, float]:
            other *= TransferFunction()
        print(other, type(other))
        if isinstance(other, TransferFunction):
            return TransferFunction(self.num*other.den, self.den*other.num)
 
    def __neg__(self): return TransferFunction(-self.num, self.den)

    def __call__(self, s):
        """
        Evaluate the transfer function for complex frequency s.

        Parameters
        ----------
        s : Complex or np.array(complex).
            Complex frequency

        Returns
        -------
        Complex or np.array(complex)
            The value(s) of the transfer function.
        """
        return self.num(s) / self.den(s)

    def bilinear_transform(self, fs=None, Ts=None, prewarp_freq=None):
        """
        Transform the s domain transfer function to the z domain.

        Parameters
        ----------
        fs : Float, optional
             Sampling frequency. The default is None.
        Ts : Float, optional
             Sampling time. The default is None.
        prewarp_freq : Float, optional
             Prewarp frequency. The default is None.

        Raises
        ------
        ValueError
            The bilinear transform can only be carried out if the sampling
            time can be determined by passing Ts or fs.

        Returns
        -------
        TransferFunction
            The discrete time transfer function in the z domain.
            This transfer function has a Ts value set.
        """
        if fs is None and Ts is None:
            raise ValueError("Must specify fs or Ts")
        Ts = Ts or 1.0/fs
        if Ts <= 0:
            info.error("bt: T must be > 0")
        if prewarp_freq is not None:
            omega = 2*PI2*prewarp_freq
            Ts = (2/omega) * np.tan(omega*Ts/2)
        self.T = Ts
        b, a = bt(self.num.coef, self.den.coef, Ts)
        return TransferFunction(b, a, Ts)

    def pzk(self):
        """
        The poles, zeros and gain of the transfer function.

        Returns
        -------
        p : List
            List of poles
        z : List
            List of zeros
        k : Float
            Gain.

        """
        k = self(0)
        p = self.den.roots()
        z = self.num.roots()
        return p, z, k

    def pzinfo(self):
        """
        Pole and zero information for pretty printing and decorating purposes.

        Returns
        -------
        result : A list of tuples (Float, Char)
                Each tuple (f, symbol) contains the frequency in Hz and:
                    - a  '+' for a pole
                    - an 'x' for a complex conjugate pole pair
                    - a  'o' for a zero
        """
        poles, zeros, k = self.pzk()
        #print(poles, zeros)
        result = []
        for pole in poles:
            i = np.imag(pole)
            if abs(i) < 0.001:
                p = -np.real(pole)
                #print("real pole", p/PI2)
            else:
                p = np.imag(pole)
                #print("complex pole", p/PI2)
            if p < 0:
                result.append((-p/PI2, "+"))
            else:
                result.append((p/PI2, "x"))
        for zero in zeros:
            i = np.imag(zero)
            if i == 0:
                z = -np.real(zero)
            else:
                z = abs(i)
            result.append((z/PI2, "o"))                
            #print("zero", z/PI2)
        return result

    def to_difference_equation(self):
        """
        The coefficients b, a of the linear difference equation that
        implements the z domain transfer function in the discrete domain.

        Returns
        -------
        b : np.array(float)
            a coefficients
        a : np.array(float)
            b coefficients
        
        The nth output sample y[n] for input x[n] is computed as:
        y[n] = sum{i=0, lb}(x[n-i] * b[lb-i]) - sum{i=1, la}(y[n-i] * a[la-i])
        """
        den_lc = self.den.coef[-1]
        num = self.num / den_lc
        den = self.den / den_lc
        n = max(len(num.coef), len(den.coef))
        b = np.zeros(n)
        a = np.zeros(n)
        b[:len(num.coef)] = num.coef[::-1]
        a[:len(den.coef)] = den.coef[::-1]
        a[0] = 1.0
        return b, a

    def tdfilter(self, x, y0=0):
        """
        Calculate the output y[i] of the discrete finite difference equation
        for input vector x.

        Parameters
        ----------
        x :  np.array(float)
             Input values.
        y0 : Float, optional
             y[0]. The default is 0. Higher order boundary conditions are
             assumed to be 0.

        Returns
        -------
        y : np.array(float)
            The discrete time domain output y.
            The length of y is identical to the length of x.
        """
        try:
            Ts = self.Ts
        except:
            info.error("Sampling time not known. Set Ts first.")
        if type(self.a) != np.ndarray:
            info.error("Digital coefficients not known. Call bt(T) first.")

        lx = len(x)
        la = len(self.a)
        lb = len(self.b)
        la1 = la - 1
        lb1 = lb - 1
        x = np.pad(x, (y0, lb))  # add some zeros for negative indices
        y = np.zeros(lx)

        for n in range(lx):
            for i in range(lb):
                y[n] += self.b[lb1-i] * x[n-i]
            for i in range(1, la):
                y[n] -= self.a[la1-i] * y[n-i]
        return y

    def ira(self, Ts, n):
        """
        The impulse response of the continous time transfer function,
        calculated through the inverse fft of the frequency response.

        Parameters
        ----------
        Ts : Float
             The time domain interval for which samples will be calculated.
        n :  Int
             The number of samples.

        Returns
        -------
        t :  n.array(float)
             Sample times
        re : np.array(float)
             Impulse response of the transfer function.
        """
        t = np.linspace(0, (n-1)*Ts, n)
        f = np.fft.rfftfreq(n, Ts)
        H = self
        w = PI2 * f
        s = 1j * w
        ha = H(s)
        ir = np.fft.irfft(ha)
        re = np.real(ir)
        return t, re

    def ird(self, Ts, n):
        """
        The impulse response of the discrete time transfer function,
        calculated through the finite difference equation.

        Parameters
        ----------
        Ts : Float
             The time domain interval for which samples will be calculated.
        n :  Int
             The number of samples.

        Returns
        -------
        t :  n.array(float)
             Sample times
        y : np.array(float)
             Impulse response of the discrete time transfer function.
        """
        t = np.linspace(0, (n-1)*Ts, n)
        x = np.zeros(len(t))
        x[0] = 1  # Impulse
        y = self.tdfilter(x)
        return t, y

    # TODO
    def step_response(self, fs=None, N=None):
        """
        The step response, calculated through convolution 

        Parameters
        ----------
        fs : TYPE, optional
            DESCRIPTION. The default is None.
        N : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        t : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """

        t, h = self.impulse_response(T_total=T_total, fs=fs, N=N)
        step = np.ones_like(t)
        y = np.convolve(h, step, mode='full')[:len(t)]
        return t, y

    def poles(self):
        """
        The poles of the transfer function, calculated from the roots of
        the denominator.

        Returns
        -------
        np.ndarray
            Array of poles.
        """
        return self.den.roots()

    def zeros(self):
        """
        The zeros of the transfer function, calculated from the roots of
        the numerator.

        Returns
        -------
        np.ndarray
            Array of zeros.
        """
        return self.num.roots()

    def __repr__(self):
        return f"TransferFunction(num={self.num}, den={self.den})"

    def __str__(self):
        n = f"{self.num}"
        d = f"{self.den}"
        m = "\n" + max(len(n), len(d)) * "-" + "\n"
        return n + m + d
