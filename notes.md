Initial transient in Gradient Tracking Method (where consensus is reached), is due to initial conditions. 

But what we really care about is the asymptotic behaviour (not transient periods).

## Analizzando Gradient Tracking Method:
What happens if we lose COLUMN stochasticity? We still have consensus, but the algo doesn't converge to the minimum of the sum of the cost functions. If you compute the left eigenvector of the matrix, than you'll see the agents are cooperatively solving ... left eigenvectors. This is going to WEIGHTED average consensus (and the weights (left eigenvector) are not 1s). 

What if we lose ROW stochasticity? Consensus is lost too, and what we reach makes no sense with respect to the optimality error. 

## Quesitons

* The field of view should apply also between agents?
* How to handle the field of view between agents?
* is the total norm of the gradient computed correctly? (we made the average between the s_k)