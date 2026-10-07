"""Ambient-before-restriction audit for the simplest response adjoint.

P_6 differentiates the ambient test, then restricts to a=z, u=0.
Differentiating the already restricted function is a different operation
when the cutoff depends on r. This module records that distinction on a
scalar model of the incidence jet.
"""

from __future__ import annotations

import sympy as sp


def ambient_then_restrict(expr: sp.Expr, a: sp.Matrix, z: sp.Matrix) -> sp.Expr:
    deriv = sp.diff(expr, a[0])
    return deriv.subs({z[i]: a[i] for i in range(4)})


def restrict_then_differentiate(expr: sp.Expr, a: sp.Matrix, z: sp.Matrix) -> sp.Expr:
    restricted = expr.subs({z[i]: a[i] for i in range(4)})
    return sp.diff(restricted, a[0])


def restriction_order_gap() -> sp.Expr:
    """Nonzero when a factor of (z-a) is differentiated before incidence.

    Restricting first kills the factor, so its derivative never appears.
    """
    a = sp.Matrix(sp.symbols("a0:4"))
    z = sp.Matrix(sp.symbols("z0:4"))
    r0 = z[0] - a[0]
    expr = r0 * sp.exp(-(r0 ** 2))
    return sp.simplify(
        ambient_then_restrict(expr, a, z) - restrict_then_differentiate(expr, a, z)
    )
