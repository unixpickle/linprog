package linprog

// Simplex runs the simplex algorithm to completion and
// returns a solution if one is found. If there is no
// solution, nil is returned and the boolean return value
// is true if the problem is unbounded, or false if the
// problem has no feasible solutions.
func Simplex(lp *StandardLP, pr PivotRule, dense bool) (Vector, bool) {
	tableau := SimplexPhase1(lp, pr, dense)
	if tableau == nil {
		return nil, false
	}
	for {
		leaving, entering, status := pr.ChoosePivot(tableau)
		if status == Unbounded {
			return nil, true
		} else if status == Optimal {
			break
		} else {
			tableau.Pivot(leaving, entering)
		}
	}
	return tableau.Solution(), true
}

// SimplexPhase1 runs phase 1 of the simplex algorithm to
// find an initial basic feasible solution.
//
// It produces a tableau for phase 2.
//
// If no basic feasible solution can be found, nil is
// returned.
func SimplexPhase1(lp *StandardLP, pr PivotRule, dense bool) *SimplexTableau {
	tableau := NewTableauPhase1(lp, dense)
	for {
		leaving, entering, status := pr.ChoosePivot(tableau)
		if status == Unbounded {
			return nil
		} else if status == Optimal {
			break
		} else {
			tableau.Pivot(leaving, entering)
		}
	}
	eps := tableau.Matrix.AbsMax() * relativeEpsilon
	if tableau.ObjectiveValue() > eps {
		return nil
	}

	if !tableau.phase1ToPhase2(lp) {
		return nil
	}
	return tableau
}
