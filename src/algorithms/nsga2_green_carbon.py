"""
NSGA-II Green Algorithm with Carbon-Aware Operators
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from src.algorithms.base import BaseAlgorithm
from src.models import CompositionProblem, Composition, Service


REGION_DISTANCES = {
    ("us-east", "us-west"): 1.5, ("us-east", "eu-west"): 2.0,
    ("us-east", "eu-north"): 2.5, ("eu-west", "eu-north"): 0.5,
    ("asia-east", "asia-south"): 1.5, ("us-west", "asia-east"): 3.0,
}


class NSGAIIGreenCarbonAware(BaseAlgorithm):
    """NSGA-II with carbon-aware initialization, crossover, and mutation."""

    def __init__(self, population_size=100, num_generations=200,
                 crossover_prob=0.9, mutation_prob=0.1,
                 geographic_repair_prob=0.5, carbon_bias_lambda=1.0,
                 distance_threshold=2.0, seed=None):
        super().__init__(population_size, num_generations, seed)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.geographic_repair_prob = geographic_repair_prob
        self.carbon_bias_lambda = carbon_bias_lambda
        self.distance_threshold = distance_threshold
        self.convergence_history = []
        self._build_distances()

    def _build_distances(self):
        self.distances = {}
        regions = ["us-east", "us-west", "eu-west", "eu-north",
                   "asia-east", "asia-south", "sa-east", "au-east"]
        for r in regions:
            self.distances[(r, r)] = 0.0
        for (r1, r2), d in REGION_DISTANCES.items():
            self.distances[(r1, r2)] = d
            self.distances[(r2, r1)] = d

    def _get_distance(self, r1, r2):
        return self.distances.get((r1, r2), 3.0)

    def _evaluate(self, encoding, problem):
        rt, carbon, cost = 0.0, 0.0, 0.0
        for i, idx in enumerate(encoding):
            s = problem.categories[i].services[int(idx)]
            rt += s.response_time
            carbon += s.carbon_emission
            cost += s.cost
        return np.array([rt, carbon, cost])

    def _create_biased_individual(self, problem, bias):
        encoding = []
        for cat in problem.categories:
            services = cat.services
            if bias == "carbon":
                sorted_svcs = sorted(services, key=lambda s: s.carbon_emission)
            elif bias == "latency":
                sorted_svcs = sorted(services, key=lambda s: s.response_time)
            else:
                sorted_svcs = sorted(services, key=lambda s: s.cost)
            if np.random.random() < 0.7:
                top_k = max(1, len(sorted_svcs) // 3)
                idx = services.index(sorted_svcs[np.random.randint(0, top_k)])
            else:
                idx = np.random.randint(0, len(services))
            encoding.append(idx)
        enc = np.array(encoding)
        return np.concatenate([enc, self._evaluate(enc, problem)])

    def _create_individual(self, problem):
        enc = np.array([np.random.randint(0, len(c.services)) for c in problem.categories])
        return np.concatenate([enc, self._evaluate(enc, problem)])

    def _carbon_aware_init(self, problem):
        """Algorithm 2: Stratified carbon-aware initialization."""
        pop = []
        q = self.population_size // 4
        for _ in range(q):
            pop.append(self._create_biased_individual(problem, "carbon"))
        for _ in range(q):
            pop.append(self._create_biased_individual(problem, "latency"))
        for _ in range(q):
            pop.append(self._create_biased_individual(problem, "cost"))
        for _ in range(self.population_size - len(pop)):
            pop.append(self._create_individual(problem))
        return np.array(pop)

    def _geographic_repair(self, encoding, problem):
        """Repair geographic incoherence."""
        enc = encoding.copy()
        n = len(enc)
        for i in range(n - 1):
            s_i = problem.categories[i].services[int(enc[i])]
            s_j = problem.categories[i + 1].services[int(enc[i + 1])]
            if self._get_distance(s_i.region, s_j.region) > self.distance_threshold:
                if np.random.random() < self.geographic_repair_prob:
                    alts = [s for s in problem.categories[i + 1].services
                            if self._get_distance(s.region, s_i.region) <= 1.5]
                    if alts:
                        best = min(alts, key=lambda s: s.carbon_emission)
                        enc[i + 1] = problem.categories[i + 1].services.index(best)
        return enc

    def _carbon_aware_crossover(self, p1, p2, problem):
        """Algorithm 3: Carbon-aware crossover with geographic repair."""
        if np.random.random() > self.crossover_prob:
            return p1.copy(), p2.copy()
        n = problem.num_categories
        e1, e2 = p1[:n].astype(int), p2[:n].astype(int)
        pts = sorted(np.random.choice(max(1, n-1), min(2, max(1, n-1)), replace=False) + 1) if n > 2 else [1]
        if len(pts) == 1:
            pts = [pts[0], n]
        c1 = np.concatenate([e1[:pts[0]], e2[pts[0]:pts[1]], e1[pts[1]:]])
        c2 = np.concatenate([e2[:pts[0]], e1[pts[0]:pts[1]], e2[pts[1]:]])
        c1 = self._geographic_repair(c1, problem)
        c2 = self._geographic_repair(c2, problem)
        return (np.concatenate([c1.astype(float), self._evaluate(c1, problem)]),
                np.concatenate([c2.astype(float), self._evaluate(c2, problem)]))

    def _carbon_biased_mutation(self, ind, problem):
        """Algorithm 4: Carbon-biased mutation."""
        n = problem.num_categories
        enc = ind[:n].copy().astype(int)
        for i in range(n):
            if np.random.random() < self.mutation_prob:
                cands = problem.categories[i].services
                curr_carbon = cands[enc[i]].carbon_emission
                weights = np.array([np.exp(-self.carbon_bias_lambda * s.carbon_emission / max(curr_carbon, 1e-10))
                                   for s in cands])
                weights /= weights.sum()
                enc[i] = np.random.choice(len(cands), p=weights)
        return np.concatenate([enc.astype(float), self._evaluate(enc, problem)])

    def _dominates(self, a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _fast_nondominated_sort(self, pop):
        n = len(pop)
        obj = pop[:, -3:]
        dom_count = np.zeros(n)
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(obj[p], obj[q]):
                    dom_set[p].append(q)
                elif self._dominates(obj[q], obj[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in dom_set[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        nxt.append(q)
            if nxt:
                fronts.append(nxt)
            i += 1
        return [f for f in fronts if f]

    def _crowding_distance(self, pop, front):
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        obj = pop[front, -3:]
        for m in range(3):
            idx = np.argsort(obj[:, m])
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            rng = obj[idx[-1], m] - obj[idx[0], m]
            if rng > 1e-10:
                for i in range(1, n - 1):
                    dist[idx[i]] += (obj[idx[i+1], m] - obj[idx[i-1], m]) / rng
        return dist

    def _tournament_select(self, pop, ranks, crowd):
        cands = np.random.choice(len(pop), 2, replace=False)
        if ranks[cands[0]] < ranks[cands[1]]:
            return pop[cands[0]].copy()
        elif ranks[cands[0]] > ranks[cands[1]]:
            return pop[cands[1]].copy()
        return pop[cands[0]].copy() if crowd[cands[0]] > crowd[cands[1]] else pop[cands[1]].copy()

    def optimize(self, problem, verbose=False):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.convergence_history = []
        pop = self._carbon_aware_init(problem)
        fronts = self._fast_nondominated_sort(pop)
        ranks = np.zeros(len(pop))
        for i, f in enumerate(fronts):
            for idx in f:
                ranks[idx] = i
        crowd = np.zeros(len(pop))
        for f in fronts:
            if f:
                crowd[f] = self._crowding_distance(pop, f)
        for gen in range(self.num_generations):
            offspring = []
            while len(offspring) < self.population_size:
                p1 = self._tournament_select(pop, ranks, crowd)
                p2 = self._tournament_select(pop, ranks, crowd)
                c1, c2 = self._carbon_aware_crossover(p1, p2, problem)
                c1 = self._carbon_biased_mutation(c1, problem)
                c2 = self._carbon_biased_mutation(c2, problem)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.population_size])
            combined = np.vstack([pop, offspring])
            fronts = self._fast_nondominated_sort(combined)
            ranks = np.zeros(len(combined))
            for i, f in enumerate(fronts):
                for idx in f:
                    ranks[idx] = i
            crowd = np.zeros(len(combined))
            for f in fronts:
                if f:
                    crowd[f] = self._crowding_distance(combined, f)
            new_pop = []
            for f in fronts:
                if len(new_pop) + len(f) <= self.population_size:
                    new_pop.extend([combined[i] for i in f])
                else:
                    rem = self.population_size - len(new_pop)
                    fc = crowd[f]
                    idx = np.argsort(-fc)[:rem]
                    new_pop.extend([combined[f[i]] for i in idx])
                    break
            pop = np.array(new_pop)
            fronts = self._fast_nondominated_sort(pop)
            ranks = np.zeros(len(pop))
            for i, f in enumerate(fronts):
                for idx in f:
                    ranks[idx] = i
            crowd = np.zeros(len(pop))
            for f in fronts:
                if f:
                    crowd[f] = self._crowding_distance(pop, f)
            pareto = self._extract_pareto(pop, problem)
            hv = self._hypervolume(pareto)
            self.convergence_history.append({"generation": gen, "hypervolume": hv, "num_pareto": len(pareto)})
            if verbose and gen % 50 == 0:
                print(f"  Gen {gen}: HV={hv:.0f}, |PF|={len(pareto)}")
        return self._extract_pareto(pop, problem)

    def _extract_pareto(self, pop, problem):
        obj = pop[:, -3:]
        pareto_idx = []
        for i in range(len(obj)):
            dominated = False
            for j in range(len(obj)):
                if i != j and self._dominates(obj[j], obj[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)
        results = []
        for i in pareto_idx:
            enc = pop[i, :problem.num_categories].astype(int)
            svcs = [problem.categories[c].services[enc[c]] for c in range(len(enc))]
            results.append(Composition(composition_id=f"sol_{i}", selected_services=svcs, objectives=obj[i]))
        return results

    def _hypervolume(self, solutions):
        if not solutions:
            return 0.0
        obj = np.array([s.objectives for s in solutions])
        ref = np.max(obj, axis=0) * 1.1 + 1.0
        sorted_sol = sorted(solutions, key=lambda s: s.response_time)
        hv = 0.0
        prev_rt = 0.0
        for sol in sorted_sol:
            w = sol.response_time - prev_rt
            h = ref[1] - sol.carbon_emission
            if w > 0 and h > 0:
                hv += w * h
            prev_rt = sol.response_time
        if sorted_sol:
            hv += max(0, ref[0] - sorted_sol[-1].response_time) * max(0, ref[1] - sorted_sol[-1].carbon_emission)
        return hv

    def get_convergence_history(self):
        return self.convergence_history
