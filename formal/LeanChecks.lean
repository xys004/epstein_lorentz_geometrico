import Std

/-
  Small exact-arithmetic checks for the manuscript's algebraic core.

  The manuscript is formulated over real smooth curves.  Lean is used here
  without Mathlib, so these checks instantiate the rational identities that do
  not require analysis, limits, or transcendental functions.  The symbolic
  real-variable checks are in formal/z3/invariants.smt2.
-/

namespace SPT

def invariant (k q D : Rat) : Rat := (k / D^2) / (q / D)^2

-- Exact unit-speed points on the tangent hemisphere.
example : (0 : Rat)^2 + 1^2 = 1 := by native_decide
example : ((3 : Rat) / 5)^2 + ((4 : Rat) / 5)^2 = 1 := by native_decide
example : ((5 : Rat) / 13)^2 + ((12 : Rat) / 13)^2 = 1 := by native_decide
example : ((12 : Rat) / 13)^2 + ((5 : Rat) / 13)^2 = 1 := by native_decide

-- Exact checks of kappa'_E = kappa_E / D^2 and q' = q / D.
example :
    invariant ((7 : Rat) / 5) ((4 : Rat) / 5) ((3 : Rat) / 2)
      = ((7 : Rat) / 5) / ((4 : Rat) / 5)^2 := by native_decide

example :
    invariant ((11 : Rat) / 9) ((12 : Rat) / 13) ((5 : Rat) / 4)
      = ((11 : Rat) / 9) / ((12 : Rat) / 13)^2 := by native_decide

example :
    invariant ((13 : Rat) / 7) ((5 : Rat) / 13) ((7 : Rat) / 6)
      = ((13 : Rat) / 7) / ((5 : Rat) / 13)^2 := by native_decide

end SPT
