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

# The tableau

During the execution of the simplex method, we maintain a matrix called the tableau. This matrix contains the equality constraints and an extra row containing the "relative cost coefficients". We use Gaussian elimination to make sure the basic variables have columns corresponding to columns of the identity. This way, the values assigned to these variables are given by the rightmost column of the tableau (which starts off as `b` from the linear program, before elimination). The relative cost coefficients, which we put in the top row above their corresponding variables' columns, indicate the rate of change of the objective function if we were to increase a non-basic variable to a non-zero value.

It was not obvious to me how the relative cost coefficients actually work and remain accurate during the course of elimination. The relative cost coefficients are updated by eliminating the cost coefficients for all of the basic variables during the execution of the simplex algorithtm, leaving only non-basic variables with non-zero relative cost coefficients. There are two points that you must understand in order to see why this procedure actually gives the correct relative cost coefficients:

 * Given a top row with all zeros for the basic variables, there is only one possible set of values for the non-basic variables. It doesn't matter if we achieve all zeros for the basic variables using elimination during the execution of the simplex method, or if we use elimination after executing several steps of the simplex method. Either way, the top row is equal to a linear combination of the constraint rows, plus the original top row, and only one linear combination of the constraint rows is capable of eliminating all the basic entries in the original top row, since there are `rank(A)` basic variables.
 * To eliminate the entry for basic variable `i` in the top row, we must have subtracted `objective[i]/A[j][i] * A[j]` from the top row, where `j` is the row containing a non-zero entry for variable `i`. This has no effect on the relative cost coefficient for the other basic variables, since `A[j]` is 0 for all other basic variables. However, it does influence the relative cost coefficient for any non-basic variable `k`, by subtracting `objective[i]/A[j][i] * A[j][k]` from the cost coefficient. In words, this corresponds to the fact that, as we increase variable `k` by some positive `delta`, in order to keep constraint `j` satisfied, we must reduce variable `i` by `delta * A[j][k] / A[j][i]`, so our objective function will change by `-objective[i] * A[j][k]/A[j][i] + objective[k]`.

# TODO

 * Figure out what happens when a basic variable can assume a zero value in the optimal case.
 * Understand why we can apply Phase I simplex to find a basic feasible solution for phase II.
 * Understand what's going on when we switch basic variables.
 * Learn about entering/leaving criteria
 * Learn about numerical stability issues (e.g. if relative cost coefficient is -epsilon)
 * Learn about cycles
