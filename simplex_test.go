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

	// Maximize 4.5x + 3.5y, subject to x-y = 1
	problem.Objective = Vector{4.5, 3.5}
	solution, ok = Simplex(problem, BlandPivotRule{})
	if solution != nil || !ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	}

	// Maximize 4.5x + 3.5y, subject to x-y = 1 and 2x-2y = 1.5.
	problem.ConstraintMatrix = &DenseMatrix{
		NumRows: 2,
		NumCols: 2,
		Data:    []float64{1, -1, 2, -2},
	}
	problem.ConstraintVector = Vector{1, 1.5}
	solution, ok = Simplex(problem, BlandPivotRule{})
	if solution != nil || ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	}

	// Maximize -4.5x + 3.5y, subject to x-y = 1 and 2x - 2y = 2.
	problem.Objective = Vector{-4.5, 3.5}
	problem.ConstraintVector = Vector{1, 2}
	solution, ok = Simplex(problem, BlandPivotRule{})
	if solution == nil || !ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	} else if !vectorsEqual(solution, Vector{1, 0}) {
		t.Errorf("unexpected solution: %v", solution)
	}
}

func TestSimplex3D(t *testing.T) {
	// Example from wikipedia: maximize 2x+3y+4z
	// subject to 3x+2y+z=10, 2x+5y+3z=15.
	problem := &StandardLP{
		Objective: Vector{2, 3, 4},
		ConstraintMatrix: &DenseMatrix{
			NumRows: 2,
			NumCols: 3,
			Data:    []float64{3, 2, 1, 2, 5, 3},
		},
		ConstraintVector: Vector{10, 15},
	}
	solution, ok := Simplex(problem, BlandPivotRule{})
	if solution == nil || !ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	} else if !vectorsEqual(solution, Vector{15.0 / 7.0, 0, 25.0 / 7.0}) {
		t.Errorf("unexpected solution: %v", solution)
	}
}

func TestSimplex6D(t *testing.T) {
	// Example from http://math.uww.edu/~mcfarlat/s-prob.htm.
	// Maximize x1 + 2x2 - x3 subject to a bunch of constraints.
	problem := &StandardLP{
		Objective: Vector{1, 2, -1, 0, 0, 0},
		ConstraintMatrix: &DenseMatrix{
			NumRows: 3,
			NumCols: 6,
			Data: []float64{
				2, 1, 1, 1, 0, 0,
				4, 2, 3, 0, 1, 0,
				2, 5, 5, 0, 0, 1,
			},
		},
		ConstraintVector: Vector{14, 28, 30},
	}
	solution, ok := Simplex(problem, BlandPivotRule{})
	if solution == nil || !ok {
		t.Errorf("unexpected return %v %v", solution, ok)
	} else if !vectorsEqual(solution, Vector{5, 4, 0, 0, 0, 0}) {
		t.Errorf("unexpected solution: %v", solution)
	}
}

func vectorsEqual(v1, v2 Vector) bool {
	if len(v1) != len(v2) {
		return false
	}
	v1 = append(Vector{}, v1...)
	v1.Add(v2, -1)
	return v1.AbsMax() < 1e-5
}
