package linprog

const relativeEpsilon = 1e-8

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

	// TODO: if there's any artificial variables in the basic
	// set, they are zero and we should swap them out for a
	// non-basic variable from the original LP.

	// TODO: create a subset of the tableau here.
}
