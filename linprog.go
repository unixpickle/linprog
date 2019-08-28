// Package linprog implements various algorithms for
// manipulating and solving linear programs.
package linprog

// A StandardLP is a linear program in standard form.
// In particular, it takes the form:
//
//     maximize c'*x subject to Ax = b
//
// Where c is the objective vector, A is the constraint
// matrix, and b is the constraint vector.
type StandardLP interface {
	Dim() int
	Objective() Vector
	ConstraintMatrix() Matrix
	ConstraintVector() Vector
}

// A Vector is an n-dimensional list of numbers.
type Vector []float64

// A Matrix is a (potentially sparse) matrix.
type Matrix interface {
	Rows() int
	Cols() int
	At(i, j int) float64
	Set(i, j int, x float64)
}
