#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 12:42:54 2026

@author: Marcel Hesselberth
"""

# tests/test_transferfunction.py
"""
Pytest suite for TransferFunction class:
- Bilinear transform correctness
- Arithmetic operations (+, -, *, /, negation)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from math import pi

from tf import TransferFunction, DEFAULTTRIM


# ──────────────────────────────────────────────── Fixtures

@pytest.fixture
def simple_lowpass():
    """1/(s + 1) continuous"""
    return TransferFunction([1.0], [1.0, 1.0])


@pytest.fixture
def integrator():
    """Pure integrator 1/s"""
    return TransferFunction([1.0], [1.0, 0.0])


@pytest.fixture
def second_order():
    """100 / (s² + 10s + 100) → ωn=10 rad/s, ζ=0.5"""
    return TransferFunction([100.0], [1.0, 10.0, 100.0])


@pytest.fixture
def tf1():
    """2 / (s + 3) continuous"""
    return TransferFunction([2.0], [1.0, 3.0])


@pytest.fixture
def tf2():
    """4 / (s + 5) continuous"""
    return TransferFunction([4.0], [1.0, 5.0])


@pytest.fixture
def tf_discrete_same():
    """Discrete example with Ts=0.1"""
    return TransferFunction([0.5, 0.5], [1.0, -0.3], Ts=0.1)


@pytest.fixture
def tf_discrete_diff_Ts():
    """Discrete with different Ts=0.05"""
    return TransferFunction([1.0, 0.2], [1.0, -0.4], Ts=0.05)


# ──────────────────────────────────────────────── Bilinear transform tests

def test_first_order_lowpass_standard(simple_lowpass):
    tf_disc = simple_lowpass.bilinear_transform(Ts=1.0)
    b, a = tf_disc.to_difference_equation(high_to_low=False)

    expected_b = np.array([1/3, 1/3])
    expected_a = np.array([-1/3, 1.0])

    assert_allclose(b, expected_b, rtol=1e-10, atol=1e-12)
    assert_allclose(a, expected_a, rtol=1e-10, atol=1e-12)
    assert tf_disc.Ts == pytest.approx(1.0)

def test_first_order_lowpass_standard(simple_lowpass):
    tf_disc = simple_lowpass.bilinear_transform(Ts=1.0)
    
    print("Calling with high_to_low=False explicitly")
    b, a = tf_disc.to_difference_equation(high_to_low=False)
    
    print("Returned b:", b)
    print("Returned a:", a)
    
    expected_b = np.array([1/3, 1/3])
    expected_a = np.array([1.0, -1/3])

    assert_allclose(b, expected_b, rtol=1e-9, atol=1e-12)
    assert_allclose(a, expected_a, rtol=1e-9, atol=1e-12)

@pytest.mark.parametrize("Ts", [0.02, 0.05, 0.1])
def test_second_order_stability(second_order, Ts):
    tf_disc = second_order.bilinear_transform(Ts=Ts)
    poles_z = tf_disc.poles()
    assert len(poles_z) == 2
    assert np.max(np.abs(poles_z)) < 1.0001, "Pole outside unit circle"


def test_prewarping(integrator):
    f_prewarp = 10.0
    tf_disc = integrator.bilinear_transform(Ts=0.01, prewarp_freq=f_prewarp)

    w = 2 * pi * f_prewarp
    s = 1j * w
    z = np.exp(s * 0.01)

    assert abs(tf_disc(z)) == pytest.approx(abs(integrator(s)), rel=0.05)


def test_dc_gain_preservation(tf1):
    dc_cont = tf1(0.0)
    tf_disc = tf1.bilinear_transform(Ts=0.05)
    assert tf_disc(1.0) == pytest.approx(dc_cont, abs=1e-10)


def test_zero_system():
    tf_zero = TransferFunction([0.0], [1.0, 1.0])
    tf_disc = tf_zero.bilinear_transform(Ts=0.1)
    b, _ = tf_disc.to_difference_equation(high_to_low=False)
    assert np.allclose(b, 0.0)


# ──────────────────────────────────────────────── Arithmetic — valid same-domain cases

def test_add_scalar(tf1):
    result = tf1 + 5.0
    assert_allclose(result.num.coef, [7.0, 15.0])   # 5(s+3) + 2
    assert_allclose(result.den.coef, [1.0, 3.0])
    assert result.Ts == 0.0


def test_add_two_continuous(tf1, tf2):
    result = tf1 + tf2
    assert_allclose(result.num.coef, [6.0, 22.0])
    assert_allclose(result.den.coef, [1.0, 8.0, 15.0])
    assert result.Ts == 0.0


def test_add_two_discrete_same_Ts(tf_discrete_same):
    tf_other = TransferFunction([1.0, 0.2], [1.0, -0.4], Ts=0.1)
    result = tf_discrete_same + tf_other
    assert result.Ts == pytest.approx(0.1)


def test_multiply_two_discrete_same_Ts(tf_discrete_same):
    tf_other = TransferFunction([1.0, 0.2], [1.0, -0.4], Ts=0.1)
    result = tf_discrete_same * tf_other
    assert result.Ts == pytest.approx(0.1)


def test_divide_two_discrete_same_Ts(tf_discrete_same):
    tf_other = TransferFunction([1.0, 0.2], [1.0, -0.4], Ts=0.1)
    result = tf_discrete_same / tf_other
    assert result.Ts == pytest.approx(0.1)


def test_negate(tf1):
    result = -tf1
    assert_allclose(result.num.coef, [-2.0])
    assert_allclose(result.den.coef, [1.0, 3.0])


# ──────────────────────────────────────────────── Arithmetic — invalid mixed / mismatched cases

def test_add_continuous_discrete_forbidden(tf1, tf_discrete_same):
    with pytest.raises(ValueError, match="continuous.*discrete"):
        _ = tf1 + tf_discrete_same


def test_add_discrete_continuous_forbidden(tf_discrete_same, tf1):
    with pytest.raises(ValueError, match="continuous.*discrete"):
        _ = tf_discrete_same + tf1


def test_subtract_mixed_forbidden(tf1, tf_discrete_same):
    with pytest.raises(ValueError, match="continuous.*discrete"):
        _ = tf1 - tf_discrete_same


def test_multiply_mixed_forbidden(tf1, tf_discrete_same):
    with pytest.raises(ValueError, match="mixed continuous / discrete"):
        _ = tf1 * tf_discrete_same


def test_divide_mixed_forbidden(tf1, tf_discrete_same):
    with pytest.raises(ValueError, match="mixed continuous / discrete"):
        _ = tf1 / tf_discrete_same


def test_rdivide_mixed_forbidden(tf_discrete_same, tf1):
    with pytest.raises(ValueError, match="mixed continuous / discrete"):
        _ = tf_discrete_same / tf1


def test_different_Ts_disallowed(tf_discrete_same, tf_discrete_diff_Ts):
    with pytest.raises(ValueError, match="different sampling times"):
        _ = tf_discrete_same + tf_discrete_diff_Ts

    with pytest.raises(ValueError, match="different sampling times"):
        _ = tf_discrete_same - tf_discrete_diff_Ts

    with pytest.raises(ValueError, match="different sampling times"):
        _ = tf_discrete_same * tf_discrete_diff_Ts

    with pytest.raises(ValueError, match="different sampling times"):
        _ = tf_discrete_same / tf_discrete_diff_Ts


def test_scalar_operations_always_allowed(tf_discrete_same):
    result_add = tf_discrete_same + 2.5
    result_mul = tf_discrete_same * 1.5
    result_div = tf_discrete_same / 0.5

    assert result_add.Ts == pytest.approx(0.1)
    assert result_mul.Ts == pytest.approx(0.1)
    assert result_div.Ts == pytest.approx(0.1)
    
    
# Polynomial coefficients tests
def test_bilinear_first_order_lowpass_coefficients():
    """
    Continuous: 1 / (s¹ + 1·s⁰)
    Coefficients: num = [1], den = [1, 1]  (power 0 first)

    Discrete (Ts=1): expected low-to-high
    num: 0.333·z⁰ + 0.333·z⁻¹
    den: 1.000·z⁰ - 0.333·z⁻¹
    """
    tf_cont = TransferFunction([1.0], [1.0, 1.0])  # 1 / (s¹ + 1·s⁰)
    tf_disc = tf_cont.bilinear_transform(Ts=1.0)

    # Expected low-to-high (power 0, power -1)
    expected_num = np.array([1/3, 1/3])      # 0.333 z⁰ + 0.333 z⁻¹
    expected_den = np.array([-1/3, 1.0])     # 1.000 z⁰ - 0.333 z⁻¹

    assert_allclose(tf_disc.num.coef, expected_num, rtol=1e-10, atol=1e-12,
                    err_msg="Numerator coefficients (power 0 → -1) mismatch")
    assert_allclose(tf_disc.den.coef, expected_den, rtol=1e-10, atol=1e-12,
                    err_msg="Denominator coefficients (power 0 → -1) mismatch")

    # DC gain check: evaluate at z = 1
    assert tf_disc(1.0) == pytest.approx(1.0, abs=1e-12)


def test_bilinear_integrator_coefficients():
    """
    Continuous: 1 / s¹  (no constant term)
    Coefficients: num = [1], den = [0, 1]  (power 0 = 0, power 1 = 1)

    Discrete (Ts=0.1): expected low-to-high
    num: 0.05·z⁰ + 0.05·z⁻¹
    den: 1.000·z⁰ - 1.000·z⁻¹
    """
    tf_cont = TransferFunction([1.0], [0.0, 1.0])  # 1 / (0·s⁰ + 1·s¹)
    tf_disc = tf_cont.bilinear_transform(Ts=0.1)

    expected_num = np.array([0.05, 0.05])    # 0.05 z⁰ + 0.05 z⁻¹
    expected_den = np.array([-1.0, 1.0])     # 1.000 z⁰ - 1.000 z⁻¹

    assert_allclose(tf_disc.num.coef, expected_num, rtol=1e-10, atol=1e-12,
                    err_msg="Integrator numerator (power 0 → -1) mismatch")
    assert_allclose(tf_disc.den.coef, expected_den, rtol=1e-10, atol=1e-12,
                    err_msg="Integrator denominator (power 0 → -1) mismatch")


def test_bilinear_second_order_coefficients():
    """100 / (s² + 10s + 100), Ts=0.1 → ωn=10 rad/s, ζ=0.5"""
    tf_cont = TransferFunction([100.0], [100.0, 10.0, 1.0])
    tf_disc = tf_cont.bilinear_transform(Ts=0.1)

    # Exact low-to-high
    expected_num = np.array([1/7, 2/7, 1/7])
    expected_den = np.array([3/7, -6/7, 1.0])

    assert_allclose(tf_disc.num.coef, expected_num, rtol=1e-8, atol=1e-12)
    assert_allclose(tf_disc.den.coef, expected_den, rtol=1e-8, atol=1e-12)
    assert tf_disc(1.0) == pytest.approx(1.0, abs=1e-10)