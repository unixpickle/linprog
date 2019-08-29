package linprog

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

// NewTableauPhase1 creates a SimplexTableau by wrapping a
// standard form linear program with artificial variables.
// The new objective will aim to set the artificial
// variables to zero, resulting in a smaller tableau for
// phase 2 of the algorithm.
func NewTableauPhase1(lp StandardLP) *SimplexTableau {
	// Construct a matrix that looks like:
	//
	// [  1 0 ... 0 ... -1 ... -1 0  ]
	// [  0 1 OBJECTIVE 0 ..... 0 0  ]
	// [  [0]    [A]       [I]    b  ]
	//
	numConstraints := len(lp.ConstraintVector())
	row1 := make(Vector, lp.Dim()+3+numConstraints)
	row1[0] = 1
	for i := lp.Dim() + 2; i < len(row1)-1; i++ {
		row1[i] = -1
	}
	row2 := append(append(Vector{0, 1}, lp.Objective()...), make(Vector, numConstraints+1)...)
	block0 := NewSparseMatrix(numConstraints, 2)
	block1 := lp.ConstraintMatrix().Copy()
	block2 := NewSparseMatrixIdentity(numConstraints)
	block3 := lp.ConstraintVector().Col().Copy()
	for i, bValue := range lp.ConstraintVector() {
		if bValue < 0 {
			block3.ScaleRow(i, -1)
			block1.ScaleRow(i, -1)
		}
	}
	matrix := RowBlockMatrix{
		row1.Col(),
		row2.Col(),
		ColumnBlockMatrix{block0, block1, block2, block3},
	}

	// Setup the objective value and relative cost
	// coefficients via very simple elimination.
	for i := 2; i < numConstraints+2; i++ {
		matrix.AddRow(i, 0, 1)
	}

	res := &SimplexTableau{
		Matrix:          matrix,
		RowToBasic:      make([]int, numConstraints+1),
		BasicToRow:      map[int]int{0: 0},
		FrozenVariables: map[int]bool{0: true},
	}
	for i := 1; i < numConstraints+1; i++ {
		basic := lp.Dim() + 1 + i
		res.BasicToRow[basic] = i
		res.RowToBasic[i] = basic
	}
	return res
}

// Dim returns the number of variables.
func (s *SimplexTableau) Dim() int {
	return s.Matrix.Cols() - 2
}

// Pivot takes two variables, one which is basic and one
// which is not, and swaps their roles.
func (s *SimplexTableau) Pivot(leaving, entering int) {
	row := s.BasicToRow[leaving] + 1
	column := entering + 1

	coeff := s.Matrix.At(row, column)
	s.Matrix.ScaleRow(row, 1/coeff)

	for i := 0; i < s.Matrix.Rows(); i++ {
		if i == row {
			continue
		}
		s.Matrix.AddRow(row, i, -s.Matrix.At(i, column))
	}

	s.RowToBasic[row] = entering
	delete(s.BasicToRow, leaving)
	s.BasicToRow[entering] = row - 1
}

// ObjectiveValue gets the current value of the objective
// function.
func (s *SimplexTableau) ObjectiveValue() float64 {
	return s.Matrix.At(0, s.Matrix.Cols()-1)
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
