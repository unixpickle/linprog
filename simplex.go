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

// A SimplexTableau stores the state of an instance of the
// simplex algorithm.
type SimplexTableau struct {
	Matrix Matrix

	// RowToBasic maps each inequality row to the basic
	// variable set for that row.
	//
	// Starts at the second row of the matrix, since the
	// first row is the objective.
	RowToBasic []int

	// BasicToRow maps basic variables to rows. This is
	// the inverse of RowToBasic.
	// If a variable is missing as a key, it is not a
	// basic variable.
	BasicToRow map[int]int

	// FrozenVariables is a set of variable indicates that
	// should not be pivoted. PivotRules must respect this
	// set.
	FrozenVariables map[int]bool
}

// Dim returns the number of variables.
func (s *SimplexTableau) Dim() int {
	return s.Matrix.Cols() - 2
}

// Pivot takes two variables, one which is basic and one
// which is not, and swaps their roles.
func (s *SimplexTableau) Pivot(basic, nonBasic int) {
	row := s.BasicToRow[basic] + 1
	column := nonBasic + 1

	coeff := s.Matrix.At(row, column)
	s.Matrix.ScaleRow(row, 1/coeff)

	for i := 0; i < s.Matrix.Rows(); i++ {
		if i == row {
			continue
		}
		s.Matrix.AddRow(row, i, -s.Matrix.At(i, column))
	}

	s.RowToBasic[row] = nonBasic
	delete(s.BasicToRow, basic)
	s.BasicToRow[nonBasic] = row - 1
}

// Cost gets the relative cost coefficient for a variable.
func (s *SimplexTableau) Cost(i int) float64 {
	return s.Matrix.At(0, i+1)
}

// Basic checks if a variable is basic.
func (s *SimplexTableau) Basic(i int) bool {
	_, res := s.BasicToRow[i]
	return res
}

// A PivotRule is a rule for determining which pivot to
// make in each iteration of the simplex method.
//
// It also indicates if the algorithm should halt.
type PivotRule interface {
	ChoosePivot(s *SimplexTableau) (basic, nonBasic int, status SimplexStatus)
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