"""Transparent source-derived A/B/C byte layouts used by the local array backbone."""
import numpy as np


def a_offset(m, k, width):
    return (16 * width * (m // 16) + 512 * (k // 32) +
            16 * (4 * (m % 8) + (k % 16) // 4) + 8 * ((k % 32) // 16) +
            4 * ((m % 16) // 8) + k % 4)


def b_offset(k, n, width):
    return (32 * width * (k // 32) + 512 * (n // 16) +
            16 * (4 * (n % 8) + (k % 16) // 4) + 8 * ((n % 16) // 8) +
            4 * ((k % 32) // 16) + k % 4)


def c_offset(m, n, width):
    return (16 * width * (m // 16) + 512 * (n // 32) +
            16 * (4 * (m % 8) + (n % 8) // 2) + 8 * ((n % 32) // 16) +
            4 * ((m % 16) // 8) + 2 * ((n % 16) // 8) + n % 2)


def pi(k):
    """Consumer MMA-A channel -> producer C channel permutation."""
    return 16 * (k // 16) + 2 * ((k % 16) // 4) + 8 * ((k % 4) // 2) + k % 2
