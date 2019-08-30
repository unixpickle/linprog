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
		if !s.Basic(i) && s.Cost(i) > 0 {
			enterVar = i
			break
		}
	}
	if enterVar == -1 {
		return 0, 0, Optimal
	}
	leaveVar := minRatioLeaveVariable(s, enterVar)
	if leaveVar == -1 {
		return 0, 0, Unbounded
	}
	return leaveVar, enterVar, Working
}

// GreedyPivotRule is a PivotRule that picks the column
// with the highest relative cost coefficient.
type GreedyPivotRule struct{}

func (g GreedyPivotRule) ChoosePivot(s *SimplexTableau) (int, int, SimplexStatus) {
	enterVar := -1
	bestCost := 0.0
	for i := 0; i < s.Dim(); i++ {
		cost := s.Cost(i)
		if !s.Basic(i) && cost > bestCost {
			enterVar = i
			bestCost = cost
		}
	}
	if enterVar == -1 {
		return 0, 0, Optimal
	}
	leaveVar := minRatioLeaveVariable(s, enterVar)
	if leaveVar == -1 {
		return 0, 0, Unbounded
	}
	return leaveVar, enterVar, Working
}

func minRatioLeaveVariable(s *SimplexTableau, enterVar int) int {
	leaveVar := -1
	minRatio := math.Inf(1)
	for row, basic := range s.RowToBasic {
		entry := s.Matrix.At(row, enterVar)
		if entry > 0 {
			ratio := s.Matrix.At(row, s.Matrix.Cols()-1) / entry
			if ratio < minRatio {
				minRatio = ratio
				leaveVar = basic
			}
		}
	}
	return leaveVar
}
