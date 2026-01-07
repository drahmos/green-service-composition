# Theoretical Analysis: Carbon-Aware Web Service Composition

This document provides the formal theoretical foundation for the GreenComp framework.

## 1. Problem Formulation

Let $S = \{C_1, C_2, \dots, C_n\}$ be a set of $n$ service categories. Each category $C_i$ contains a set of candidate services $s_{i,j}$. Each service $s_{i,j}$ is characterized by a tuple of QoS and environmental attributes:
$$s_{i,j} = \langle t_{i,j}, c_{i,j}, e_{i,j}, \phi_{i,j}, r_{i,j} \rangle$$
where:
- $t_{i,j}$: Response time
- $c_{i,j}$: Execution cost
- $e_{i,j}$: Energy consumption
- $\phi_{i,j}$: Carbon intensity of the data center region
- $r_{i,j}$: Availability

The carbon footprint of service $s_{i,j}$ is $f_{i,j} = e_{i,j} \times \phi_{i,j}$.

The goal is to select exactly one service from each category to form a composition $W = \{s_1, s_2, \dots, s_n\}$ that minimizes the multi-objective vector:
$$F(W) = [\sum t_i, \sum f_i, \sum c_i]$$
subject to:
- $\sum t_i \leq T_{max}$
- $\prod r_i \geq R_{min}$
- $\sum c_i \leq C_{max}$

## 2. NP-Hardness Proof

**Theorem 1**: The Carbon-Aware Web Service Composition (CA-WSC) problem is NP-hard.

**Proof**: We reduce the Multi-Objective Knapsack Problem (MOKP), which is known to be NP-hard, to CA-WSC.
Consider a simplified version of CA-WSC with two objectives (Response Time and Carbon Footprint) and one constraint (Total Cost).
Each service category $C_i$ corresponds to an item choice in the knapsack. Selecting service $s_{i,j}$ corresponds to picking a specific version of item $i$ with weight $c_{i,j}$ and multi-dimensional values $t_{i,j}$ and $f_{i,j}$. 
Since the Multiple-Choice Multi-Objective Knapsack Problem is a generalization of the classic Knapsack problem and is NP-hard, and CA-WSC can directly model it, CA-WSC is also NP-hard. 

## 3. Complexity of NSGA-II-Green

The computational complexity of one generation of NSGA-II-Green is dominated by the non-dominated sorting and crowding distance calculation.

For a population size $N$ and $M$ objectives:
1. **Non-dominated Sort**: $O(MN^2)$ using the fast non-dominated sort algorithm.
2. **Crowding Distance**: $O(MN \log N)$ for sorting the population in each objective space.
3. **Genetic Operators**: $O(N \times n)$ where $n$ is the number of categories.

Total complexity per generation: $O(MN^2)$.
Across $G$ generations: $O(GMN^2)$.

For our default parameters ($N=100, M=3, G=200$), the total operations are roughly $200 \times 3 \times 100^2 = 6,000,000$, which is well within the capabilities of modern CPU resources (as verified in our Colab performance assessment).

## 4. Convergence and Approximation

NSGA-II-Green uses an elitist strategy that guarantees that the best-found Pareto front never regresses. Under the assumption of sufficient mutation and crossover exploration, the algorithm converges to the true Pareto front $PF^*$ as $G \to \infty$. 

In practical execution, the **Hypervolume (HV)** indicator is used to measure the convergence quality:
$$HV(P, r) = \text{volume}(\bigcup_{x \in P} [x, r])$$
where $P$ is the obtained Pareto set and $r$ is the reference point. Our framework ensures a monotonic increase in expected HV over generations.
