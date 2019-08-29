package linprog

import "testing"

func TestSimplex2D(t *testing.T) {
	// Maximize -4.5x + 3.5y, subject to x-y = 1
	problem := &StandardLP{
		Objective: Vector{-4.5, 3.5},
		ConstraintMatrix: &DenseMatrix{
			NumRows: 1,
			NumCols: 2,
			Data:    []float64{1, -1},
		},
		ConstraintVector: Vector{1},
	}
	solution, ok := Simplex(problem, BlandPivotRule{})
	if solution == nil || !ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	} else if !vectorsEqual(solution, Vector{1, 0}) {
		t.Errorf("unexpected solution: %v", solution)
	}
}

func vectorsEqual(v1, v2 Vector) bool {
	v1 = append(Vector{}, v1...)
	v1.Add(v2, -1)
	return v1.AbsMax() < 1e-5
}
