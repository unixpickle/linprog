package linprog

// A SimplexTableau stores the state of an instance of the
// simplex algorithm.
//
// The tableau is a compact and slightly revised version
// of the one on wikipedia. It looks like so:
//
//     [ A b ]
//     [ c z ]
//
// Where b is the column of basic values, A is the
// constraint matrix, c is the relative cost coefficients,
// and z is the current objective value scalar.
type SimplexTableau struct {
	Matrix Matrix

	// RowToBasic maps each inequality row to the basic
	// variable set for that row.
	//
	// If a row is missing, it is because that row was all
	// zeros and needn't be used.
	RowToBasic map[int]int

	// BasicToRow maps basic variables to rows. This is
	// the inverse of RowToBasic.
	//
	// If a variable is missing, it is non-basic.
	BasicToRow map[int]int
}

// NewTableauPhase1 creates a SimplexTableau by wrapping a
// standard form linear program with artificial variables.
// The new objective will aim to set the artificial
// variables to zero, resulting in a smaller tableau for
// phase 2 of the algorithm.
func NewTableauPhase1(lp StandardLP) *SimplexTableau {
	// Construct a matrix that looks like:
	//
	// [    [A]       [I]    b ]
	// [ ... 0 ... -1 ... -1 0 ]
	//
	numConstraints := len(lp.ConstraintVector())
	lastRow := make(Vector, lp.Dim()+numConstraints+1)
	for i := lp.Dim(); i < len(lastRow)-1; i++ {
		lastRow[i] = -1
	}
	block1 := lp.ConstraintMatrix().Copy()
	block2 := NewSparseMatrixIdentity(numConstraints)
	block3 := lp.ConstraintVector().Col().Copy()
	for i, bValue := range lp.ConstraintVector() {
		if bValue < 0 {
			block1.ScaleRow(i, -1)
			block3.ScaleRow(i, -1)
		}
	}
	matrix := RowBlockMatrix{
		ColumnBlockMatrix{block1, block2, block3},
		lastRow.Row(),
	}

	// Setup the objective value and relative cost
	// coefficients via very simple elimination.
	for i := 0; i < numConstraints; i++ {
		matrix.AddRow(i, numConstraints, 1)
	}

	res := &SimplexTableau{
		Matrix:     matrix,
		RowToBasic: map[int]int{},
		BasicToRow: map[int]int{},
	}
	for i := 0; i < numConstraints; i++ {
		basic := lp.Dim() + i
		res.BasicToRow[basic] = i
		res.RowToBasic[i] = basic
	}
	return res
}

// Dim returns the number of variables.
func (s *SimplexTableau) Dim() int {
	return s.Matrix.Cols() - 1
}

// Pivot takes two variables, one which is basic and one
// which is not, and swaps their roles.
func (s *SimplexTableau) Pivot(leaving, entering int) {
	row := s.BasicToRow[leaving]
	column := entering

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
	s.BasicToRow[entering] = row
}

// ObjectiveValue gets the current value of the objective
// function.
func (s *SimplexTableau) ObjectiveValue() float64 {
	return s.Matrix.At(s.Matrix.Rows()-1, s.Matrix.Cols()-1)
}

// Cost gets the relative cost coefficient for a variable.
func (s *SimplexTableau) Cost(i int) float64 {
	return s.Matrix.At(s.Matrix.Rows()-1, i)
}

// Basic checks if a variable is basic.
func (s *SimplexTableau) Basic(i int) bool {
	_, res := s.BasicToRow[i]
	return res
}

func (s *SimplexTableau) phase1ToPhase2(lp StandardLP) *SimplexTableau {
	// TODO: this.
	return nil
}
