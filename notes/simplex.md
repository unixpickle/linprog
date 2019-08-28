# Simplex Method

This document contains my notes about the simplex method. The wikipedia page about the simplex method leaves a lot to be desired, especially in the way of explanations and proofs.

# Basic feasible solutions

A **basic feasible solution** is a point on the convex polytope defined by an LP's linear constraints. The simplex algorithm finds basic feasible solutions by choosing `rank(A)` variables to be "basic", and then setting the remaining non-basic variables to zero. Therefore, it must be shown that there is a one-to-one correspondence between vertices on the polytope and solutions with exactly `rank(A)` basic variables. This is far from obvious to me.

For a point `x` to be a vertex on the polytope, it must have two properties:

 * The point `x` must satisfy the constraints.
 * There must be no non-zero vector `v` for which `x + delta*v` and `x - delta*v` both satisfy the constraints when `delta` is arbitrarily small. Otherwise, we are on an edge/side, or inside of the polytope.

Now, we will prove that all solutions where there are `rank(A)` basic variables are vertices of the polytope. The first condition is always going to be met (i.e. we will not be violating the constraints), so we simply need to check the second condition. We will break down this condition into two cases:

 * Suppose vector `v` has non-zero values for the non-basic variables. Then one of the two directions along `v` will cause the non-basic variable to go negative, breaking the constraints.
 * Otherwise, vector `v` is only non-zero for the basic variables. Since there are exactly `rank(A)` basic variables, the basic variables can only assume one possible value to satisfy the equality constraints (since it is a linear system of `rank(A)` variables and `rank(A)` equations).

Next, we must prove that all vertices of the polytope are solutions with `rank(A)` basic variables. Suppose we have a feasible solution `x` with more than `rank(A)` non-zero variables. Then the equality constraints are under-determined. Let `M` be the matrix obtained by taking the columns of `A` corresponding to the non-zero variables, and let `z` be the vector obtained by taking the values of the non-basic variables from `x`. Since we assume that the equality constraints are currently satisfied, we know that `M*z = b`. Since `rank(M) <= rank(A)`, and `dim(z) > rank(A)`, there is some vector `v` in the null-space of `M` (so `Av = 0`). We can clearly see that, for sufficiently small `delta`, both `z + v >= 0` and `z - v >= 0`, since `z > 0`. Further, `A(z+delta*v) = Az + 0 = b`, so the equality constraints are still met. Therefore, `x` is on an edge or side, or is fully inside the polytope.