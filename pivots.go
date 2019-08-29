package linprog

import "math"

// SimplexStatus is the status of an instance of the
// simplex algorithm.
type SimplexStatus int

const (
	// Working indicates that the simplex algorithm still
	// has work to do.
	Working SimplexStatus = iota

	// Optimal indicates that an optimum has been found.
	Optimal

	// Unbounded indicates that the objective is unbounded
	// and an infinitely large value can be achieved.
	Unbounded
)

// A PivotRule is a rule for determining which pivot to
// make in each iteration of the simplex method.
//
// It also indicates if the algorithm should halt.
type PivotRule interface {
	ChoosePivot(s *SimplexTableau) (leaving, entering int, status SimplexStatus)
}

// BlandPivotRule is a PivotRule that implements Bland's
// rule for avoiding cycles in the simplex method.
type BlandPivotRule struct{}

func (b BlandPivotRule) ChoosePivot(s *SimplexTableau) (int, int, SimplexStatus) {
	enterVar := -1
	for i := 0; i < s.Dim(); i++ {
		if !s.Basic(i) && !s.FrozenVariables[i] && s.Cost(i) > 0 {
			enterVar = i
			break
		}
	}
	if enterVar == -1 {
		return 0, 0, Optimal
	}
	pivotRow := -1
	minRatio := math.Inf(1)
	for i := 1; i < s.Matrix.Rows(); i++ {
		if s.FrozenVariables[s.RowToBasic[i-1]] {
			continue
		}
		entry := s.Matrix.At(enterVar+1, i)
		if entry > 0 {
			ratio := s.Matrix.At(s.Matrix.Cols()-1, i) / entry
			if ratio < minRatio {
				minRatio = ratio
				pivotRow = i
			}
		}
	}
	if pivotRow == -1 {
		return 0, 0, Unbounded
	}
	leaveVar := s.RowToBasic[pivotRow-1]

	return leaveVar, enterVar, Working
}
