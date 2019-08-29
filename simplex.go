package linprog

// SimplexPhase1 runs phase 1 of the simplex algorithm to
// find an initial basic feasible solution.
//
// It produces a tableau for phase 2.
//
// If no basic feasible solution can be found, nil is
// returned.
func SimplexPhase1(lp StandardLP, pr PivotRule) *SimplexTableau {
	tableau := NewTableauPhase1(lp)
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
