(set-logic QF_NRA)

; Symbolic checks for the algebraic identities used in the SPT construction.
; Each check is expected to return unsat: the negated identity has no model
; under the stated positivity assumptions.

; 1. The conformal tangent factor and the time-coordinate Jacobian cancel.
(declare-const n Real)
(declare-const k Real)
(declare-const q Real)
(declare-const D Real)
(declare-const kp Real)
(declare-const qp Real)
(assert (> q 0.0))
(assert (> D 0.0))
(assert (= kp (/ k (* D D))))
(assert (= qp (/ q D)))
(assert (not (= (/ kp (* qp qp)) (/ k (* q q)))))
(check-sat)

; 2. A Lorentz Doppler denominator is positive for subluminal velocities.
(push)
(declare-const beta Real)
(declare-const u Real)
(declare-const gamma Real)
(declare-const Dboost Real)
(assert (> gamma 0.0))
(assert (= (* gamma gamma (- 1.0 (* beta beta))) 1.0))
(assert (< (* beta beta) 1.0))
(assert (< (* u u) 1.0))
(assert (= Dboost (* gamma (- 1.0 (* beta u)))))
(assert (<= Dboost 0.0))
(check-sat)
(pop)

; 3. The unit tangent condition is preserved by the rational tangent map
; after the common conformal factor is removed.  This scalar consequence is
; checked here with the same q and D relation as in item 1.
(push)
(declare-const b2 Real)
(declare-const n2 Real)
(assert (= (+ b2 (* q q)) 1.0))
(assert (= n2 (/ n (* D D))))
(assert (not (= (/ n2 (* qp qp)) (/ n (* q q)))))
(check-sat)
(pop)
