"""
:mod:`optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .quickprop_sgd import QuickPropSGD
from .ellipsoid_sgd import EllipsoidSGD
del ellipsoid_sgd
del quickprop_sgd
