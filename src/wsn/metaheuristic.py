"""Heurística constructiva Greedy balanceada y Recocido Simulado (SA) para WSN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

from wsn.instance import WSNInstance


@dataclass
class MetaheuristicResult:
    """Resultado de resolución mediante heurística constructiva o metaheurística."""

    algorithm: str
    mode: str
    topology: dict[int, int]
    edges: list[tuple[int, int]]
    flows: dict[tuple[int, int], float]
    objective: float
    max_energy_uJ: float
    total_energy_uJ: float
    lifetime_rounds: float
    solve_time_s: float
    iterations: int
    accepted_moves: int
    improving_moves: int
    seed: int | None
    convergence_history: list[dict[str, float | int]]


class _TreeState:
    """Representación incremental de árbol enraizado en el Gateway (nodo 0)."""

    def __init__(
        self,
        instance: WSNInstance,
        parent_map: dict[int, int],
        mode: str = "extended",
    ) -> None:
        self.instance = instance
        self.mode = mode
        self.n = instance.N
        self.k_limit: int | None = None if mode == "base" else instance.K_max
        self.tx_cost = instance.tx_cost_matrix(mode=mode)
        self.e_rx = float(instance.e_rx)
        self.g = instance.g.astype(np.float64, copy=True)
        self.eff_battery = (
            np.ones(self.n, dtype=np.float64)
            if mode == "base"
            else instance.battery_levels.astype(np.float64, copy=True)
        )

        adj = instance.neighbors_out()
        self.neighbors_out: list[list[int]] = [[]] + [
            sorted(adj[i], key=lambda p: (float(self.tx_cost[i, p]), p))
            for i in range(1, self.n)
        ]
        self.neighbor_sets: list[set[int]] = [set(lst) for lst in self.neighbors_out]

        self.parent = np.full(self.n, -1, dtype=np.int64)
        for i in range(1, self.n):
            self.parent[i] = int(parent_map[i])

        self.children: list[set[int]] = [set() for _ in range(self.n)]
        self.fan_in = np.zeros(self.n, dtype=np.int64)
        self.subtree_traffic = np.zeros(self.n, dtype=np.float64)
        self.node_energy = np.zeros(self.n, dtype=np.float64)
        self.node_rate = np.zeros(self.n, dtype=np.float64)
        self.recompute_all()

    def recompute_all(self) -> None:
        """Recalcula hijos, fan-in, tráfico, energías y tasas desde cero."""
        self.children = [set() for _ in range(self.n)]
        self.fan_in.fill(0)
        for i in range(1, self.n):
            p = int(self.parent[i])
            self.children[p].add(i)
            self.fan_in[p] += 1

        # Orden BFS desde el nodo 0, luego procesar en reverso (hojas hacia raíz)
        order: list[int] = []
        queue = deque([0])
        while queue:
            u = queue.popleft()
            order.append(u)
            for c in self.children[u]:
                queue.append(c)

        self.subtree_traffic.fill(0.0)
        self.node_energy.fill(0.0)
        self.node_rate.fill(0.0)

        for u in reversed(order):
            if u == 0:
                continue
            in_flow = sum(float(self.subtree_traffic[c]) for c in self.children[u])
            out_flow = in_flow + float(self.g[u])
            p = int(self.parent[u])
            e_u = self.e_rx * in_flow + float(self.tx_cost[u, p]) * out_flow
            self.subtree_traffic[u] = out_flow
            self.node_energy[u] = e_u
            self.node_rate[u] = e_u / float(self.eff_battery[u])

    def alpha(self) -> float:
        return float(np.max(self.node_rate[1:]))

    def total_energy(self) -> float:
        return float(np.sum(self.node_energy[1:]))

    def guide_score(self) -> float:
        """Objetivo suave que desempata mesetas Min-Max usando RMS y media de desgaste."""
        rates = self.node_rate[1:]
        mx = float(np.max(rates))
        rms = float(np.sqrt(np.mean(rates * rates)))
        mean_r = float(np.mean(rates))
        return mx + 1e-3 * rms + 1e-4 * mean_r

    def creates_cycle(self, u: int, new_parent: int) -> bool:
        """Verifica en O(profundidad) si asignar parent[u] = new_parent crea un ciclo."""
        curr = new_parent
        while curr != 0:
            if curr == u:
                return True
            curr = int(self.parent[curr])
        return False

    def can_reassign(self, u: int, new_parent: int) -> bool:
        old_p = int(self.parent[u])
        if new_parent == old_p or new_parent == u:
            return False
        if new_parent not in self.neighbor_sets[u]:
            return False
        if new_parent != 0 and self.k_limit is not None and self.fan_in[new_parent] >= self.k_limit:
            return False
        return not self.creates_cycle(u, new_parent)

    def _move_unchecked(self, u: int, new_parent: int) -> int:
        """Reasigna parent[u] -> new_parent actualizando incrementalmente ancestros."""
        old_parent = int(self.parent[u])
        delta_flow = float(self.subtree_traffic[u])

        # Cambio de energía TX propio de u
        self.node_energy[u] += (
            float(self.tx_cost[u, new_parent]) - float(self.tx_cost[u, old_parent])
        ) * delta_flow
        self.node_rate[u] = self.node_energy[u] / float(self.eff_battery[u])

        # Descontar flujo a lo largo de la ruta antigua hasta el Gateway
        curr = old_parent
        while curr != 0:
            p_curr = int(self.parent[curr])
            self.subtree_traffic[curr] -= delta_flow
            self.node_energy[curr] -= (self.e_rx + float(self.tx_cost[curr, p_curr])) * delta_flow
            self.node_rate[curr] = self.node_energy[curr] / float(self.eff_battery[curr])
            curr = p_curr

        # Sumar flujo a lo largo de la nueva ruta hasta el Gateway
        curr = new_parent
        while curr != 0:
            p_curr = int(self.parent[curr])
            self.subtree_traffic[curr] += delta_flow
            self.node_energy[curr] += (self.e_rx + float(self.tx_cost[curr, p_curr])) * delta_flow
            self.node_rate[curr] = self.node_energy[curr] / float(self.eff_battery[curr])
            curr = p_curr

        self.children[old_parent].remove(u)
        self.children[new_parent].add(u)
        self.fan_in[old_parent] -= 1
        self.fan_in[new_parent] += 1
        self.parent[u] = new_parent
        return old_parent

    def can_swap(self, u: int, v: int) -> bool:
        """Verifica si u y v pueden intercambiar padres preservando factibilidad y aciclicidad."""
        if u == v:
            return False
        pu = int(self.parent[u])
        pv = int(self.parent[v])
        if pu == pv or pu == v or pv == u:
            return False
        if pv not in self.neighbor_sets[u] or pu not in self.neighbor_sets[v]:
            return False
        # Verificar que ni pu ni pv estén en el subárbol de v o u respectivamente
        if self.creates_cycle(u, pv) or self.creates_cycle(v, pu):
            return False
        # Verificar que tras u->pv no haya ciclo en v->pu ni viceversa
        curr = pv
        while curr != 0:
            if curr == v:
                return False
            curr = int(self.parent[curr])
        curr = pu
        while curr != 0:
            if curr == u:
                return False
            curr = int(self.parent[curr])
        return True

    def swap_parents(self, u: int, v: int) -> tuple[int, int]:
        """Intercambia los padres de u y v manteniendo intacto el fan-in de todos los nodos."""
        pu = int(self.parent[u])
        pv = int(self.parent[v])
        self._move_unchecked(u, pv)
        self._move_unchecked(v, pu)
        return pu, pv

    def revert_swap(self, u: int, v: int, orig_pu: int, orig_pv: int) -> None:
        self._move_unchecked(v, orig_pv)
        self._move_unchecked(u, orig_pu)

    def subtree_nodes(self, root: int) -> list[int]:
        """Retorna todos los nodos en el subárbol de `root` (incluyéndolo)."""
        nodes: list[int] = []
        stack = [root]
        while stack:
            curr = stack.pop()
            nodes.append(curr)
            stack.extend(self.children[curr])
        return nodes

    def snapshot_parent(self) -> np.ndarray:
        return self.parent.copy()

    def restore_parent(self, snap: np.ndarray) -> None:
        self.parent[:] = snap
        self.recompute_all()

    def to_topology_and_flows(self) -> tuple[dict[int, int], list[tuple[int, int]], dict[tuple[int, int], float]]:
        topology = {i: int(self.parent[i]) for i in range(1, self.n)}
        edges = sorted((i, p) for i, p in topology.items())
        flows = {(i, p): float(self.subtree_traffic[i]) for i, p in edges}
        return topology, edges, flows


def _dijkstra_energy_order(
    instance: WSNInstance,
    mode: str,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Calcula distancias ponderadas por energía/batería desde el Gateway (nodo 0)."""
    n = instance.N
    tx_cost = instance.tx_cost_matrix(mode=mode)
    e_rx = float(instance.e_rx)
    eff_bat = (
        np.ones(n, dtype=np.float64)
        if mode == "base"
        else instance.battery_levels.astype(np.float64)
    )
    adj = instance.neighbors_out()
    rev_adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u in range(1, n):
        for p in adj[u]:
            rev_adj[p].append(u)

    dist = np.full(n, math.inf, dtype=np.float64)
    dist[0] = 0.0
    pq: list[tuple[float, int]] = [(0.0, 0)]
    while pq:
        d_p, p = heapq.heappop(pq)
        if d_p > dist[p]:
            continue
        for u in rev_adj[p]:
            step_cost = float(tx_cost[u, p]) / float(eff_bat[u])
            if p != 0:
                step_cost += e_rx / float(eff_bat[p])
            cand = d_p + step_cost
            if cand < dist[u]:
                dist[u] = cand
                heapq.heappush(pq, (cand, u))
    return dist, adj


def _remaining_nodes_can_connect(
    unassigned: set[int],
    exclude_u: int,
    in_tree_after: set[int],
    fan_in: np.ndarray,
    saturated_p: int,
    k_limit: int,
    adj: dict[int, list[int]],
) -> bool:
    """Verifica mediante BFS si saturar `saturated_p` deja algún nodo pendiente aislado."""
    rem = unassigned - {exclude_u}
    if not rem:
        return True
    seeds = [
        t
        for t in in_tree_after
        if t == 0 or (t != saturated_p and int(fan_in[t]) < k_limit)
    ]
    if not seeds:
        return False
    visited = set(seeds)
    queue = deque(seeds)
    # Grafo inverso rápido sobre nodos pendientes
    while queue:
        curr = queue.popleft()
        for nxt in rem:
            if nxt not in visited and curr in adj[nxt]:
                visited.add(nxt)
                queue.append(nxt)
    return rem.issubset(visited)


def solve_greedy(
    instance: WSNInstance,
    mode: str = "extended",
) -> MetaheuristicResult:
    """Heurística constructiva Greedy balanceada por energía, carga y fan-in.

    Crece un árbol enraizado en el nodo 0 priorizando cercanía energética (Dijkstra) y
    seleccionando para cada sensor el padre disponible que minimiza el pico resultante
    de tasa de agotamiento alpha = max_i E_i / B_i a lo largo de la ruta hasta el Gateway.
    """
    if mode not in ("base", "extended"):
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    t0 = time.perf_counter()
    n = instance.N
    k_limit: int | None = None if mode == "base" else instance.K_max
    tx_cost = instance.tx_cost_matrix(mode=mode)
    e_rx = float(instance.e_rx)
    g = instance.g
    eff_bat = (
        np.ones(n, dtype=np.float64)
        if mode == "base"
        else instance.battery_levels.astype(np.float64)
    )

    dijkstra_dist, adj = _dijkstra_energy_order(instance, mode=mode)
    adj_sets = {i: set(adj[i]) for i in range(1, n)}

    parent: dict[int, int] = {}
    fan_in = np.zeros(n, dtype=np.int64)
    flow_out = np.zeros(n, dtype=np.float64)
    node_energy = np.zeros(n, dtype=np.float64)
    node_rate = np.zeros(n, dtype=np.float64)

    in_tree: set[int] = {0}
    unassigned: set[int] = set(range(1, n))
    current_alpha = 0.0
    fallback_used = False

    while unassigned:
        # Candidatos de frontera: sensores no asignados con al menos 1 padre con cupo en el árbol
        frontier_options: list[tuple[int, list[int]]] = []
        for u in unassigned:
            valid_p = [
                p
                for p in adj[u]
                if p in in_tree and (p == 0 or k_limit is None or int(fan_in[p]) < k_limit)
            ]
            if valid_p:
                frontier_options.append((u, valid_p))

        if not frontier_options:
            # Respaldo determinista garantizado mediante BFS capacitado
            fallback_tree = instance.build_capacitated_tree(k_max=k_limit)
            if fallback_tree is None:
                raise RuntimeError("No existe árbol factible hacia el Gateway para esta instancia.")
            parent = fallback_tree
            fallback_used = True
            break

        # Priorizar sensores con menor holgura de padres y menor distancia energética al Gateway
        frontier_options.sort(
            key=lambda item: (
                0 if len(item[1]) == 1 else 1,
                float(dijkstra_dist[item[0]]),
                len(item[1]),
                item[0],
            )
        )
        u, valid_parents = frontier_options[0]

        best_p: int | None = None
        best_score: tuple[float, float, float, float] | None = None
        g_u = float(g[u])

        for p in valid_parents:
            # Si saturar p corta la conectividad para los restantes, descartar p si hay alternativas
            if (
                k_limit is not None
                and p != 0
                and int(fan_in[p]) + 1 == k_limit
                and len(valid_parents) > 1
            ):
                if not _remaining_nodes_can_connect(
                    unassigned=unassigned,
                    exclude_u=u,
                    in_tree_after=in_tree | {u},
                    fan_in=fan_in,
                    saturated_p=p,
                    k_limit=k_limit,
                    adj=adj,
                ):
                    continue

            u_rate = (float(tx_cost[u, p]) * g_u) / float(eff_bat[u])
            path_max_rate = u_rate
            path_sum_sq = u_rate * u_rate
            energy_delta = float(tx_cost[u, p]) * g_u

            curr = p
            while curr != 0:
                p_curr = parent[curr]
                delta_e = (e_rx + float(tx_cost[curr, p_curr])) * g_u
                new_rate = (float(node_energy[curr]) + delta_e) / float(eff_bat[curr])
                if new_rate > path_max_rate:
                    path_max_rate = new_rate
                path_sum_sq += new_rate * new_rate
                energy_delta += delta_e
                curr = p_curr

            resulting_alpha = max(current_alpha, path_max_rate)
            score = (resulting_alpha, path_max_rate, path_sum_sq, energy_delta)
            if best_score is None or score < best_score:
                best_score = score
                best_p = p

        if best_p is None:
            best_p = min(valid_parents, key=lambda p: (float(tx_cost[u, p]), p))

        # Aplicar asignación u -> best_p y actualizar ruta hasta 0
        parent[u] = best_p
        fan_in[best_p] += 1
        flow_out[u] = g_u
        node_energy[u] = float(tx_cost[u, best_p]) * g_u
        node_rate[u] = node_energy[u] / float(eff_bat[u])
        if node_rate[u] > current_alpha:
            current_alpha = float(node_rate[u])

        curr = best_p
        while curr != 0:
            p_curr = parent[curr]
            delta_e = (e_rx + float(tx_cost[curr, p_curr])) * g_u
            flow_out[curr] += g_u
            node_energy[curr] += delta_e
            node_rate[curr] = node_energy[curr] / float(eff_bat[curr])
            if node_rate[curr] > current_alpha:
                current_alpha = float(node_rate[curr])
            curr = p_curr

        in_tree.add(u)
        unassigned.remove(u)

    state = _TreeState(instance, parent, mode=mode)
    # En caso de fallback por K_max estrecho, validar vecindarios y dejar estructura limpia
    _ = fallback_used
    _ = adj_sets
    topology, edges, flows = state.to_topology_and_flows()
    obj = state.alpha()
    max_e = float(np.max(state.node_energy[1:]))
    tot_e = state.total_energy()
    lifetime = float(instance.nominal_battery_uJ) / obj if obj > 0.0 else float("inf")
    elapsed = time.perf_counter() - t0

    return MetaheuristicResult(
        algorithm="greedy",
        mode=mode,
        topology=topology,
        edges=edges,
        flows=flows,
        objective=obj,
        max_energy_uJ=max_e,
        total_energy_uJ=tot_e,
        lifetime_rounds=lifetime,
        solve_time_s=float(elapsed),
        iterations=n - 1,
        accepted_moves=n - 1,
        improving_moves=n - 1,
        seed=None,
        convergence_history=[
            {
                "iteration": 0,
                "current_obj": obj,
                "best_obj": obj,
                "temperature": 0.0,
            }
        ],
    )


def _local_polish(state: _TreeState, max_passes: int = 30) -> int:
    """Pulido local determinista enfocado en el subárbol del nodo cuello de botella."""
    improvements = 0
    for _ in range(max_passes):
        improved = False
        cur_alpha = state.alpha()
        cur_tot = state.total_energy()
        bottleneck = int(np.argmax(state.node_rate[1:]) + 1)
        sub_nodes = state.subtree_nodes(bottleneck)
        outside_nodes = [i for i in range(1, state.n) if i not in set(sub_nodes)]

        # 1) Probar reasignación de nodos del subárbol cuello de botella
        for u in sub_nodes:
            for p_new in state.neighbors_out[u]:
                if not state.can_reassign(u, p_new):
                    continue
                old_p = state._move_unchecked(u, p_new)
                new_alpha = state.alpha()
                new_tot = state.total_energy()
                if new_alpha < cur_alpha - 1e-9 or (
                    abs(new_alpha - cur_alpha) <= 1e-9 and new_tot < cur_tot - 1e-6
                ):
                    cur_alpha = new_alpha
                    cur_tot = new_tot
                    improvements += 1
                    improved = True
                    break
                state._move_unchecked(u, old_p)
            if improved:
                break

        if improved:
            continue

        # 2) Probar intercambios 2-nodos preservando fan-in si K_max está activo
        if outside_nodes:
            for u in sub_nodes:
                for v in outside_nodes:
                    if not state.can_swap(u, v):
                        continue
                    pu, pv = state.swap_parents(u, v)
                    new_alpha = state.alpha()
                    new_tot = state.total_energy()
                    if new_alpha < cur_alpha - 1e-9 or (
                        abs(new_alpha - cur_alpha) <= 1e-9 and new_tot < cur_tot - 1e-6
                    ):
                        improvements += 1
                        improved = True
                        break
                    state.revert_swap(u, v, pu, pv)
                if improved:
                    break

        if not improved:
            break
    return improvements


def solve_simulated_annealing(
    instance: WSNInstance,
    mode: str = "extended",
    seed: int = 42,
    max_iterations: int | None = None,
    t_initial: float | None = None,
    t_min_ratio: float = 1e-4,
    cooling_rate: float | None = None,
    focus_prob: float = 0.70,
    swap_prob: float = 0.25,
    sample_every: int | None = None,
) -> MetaheuristicResult:
    """Resuelve el enrutamiento WSN mediante Recocido Simulado (Simulated Annealing).

    Parte desde la solución Greedy balanceada y aplica movimientos de vecindario sobre
    árboles dirigidos enraizados en el nodo 0 (reasignación de padre e intercambio
    2-nodos preservando fan-in), con foco probabilístico en el nodo cuello de botella
    y su subárbol para equilibrar la carga energética.
    """
    if mode not in ("base", "extended"):
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    t0_wall = time.perf_counter()
    greedy_init = solve_greedy(instance, mode=mode)
    state = _TreeState(instance, greedy_init.topology, mode=mode)
    rng = np.random.RandomState(seed)

    n = instance.N
    if max_iterations is None:
        max_iterations = max(4000, 75 * n)
    if sample_every is None:
        sample_every = max(1, max_iterations // 250)

    current_alpha = state.alpha()
    current_guide = state.guide_score()
    current_tot_e = state.total_energy()

    best_alpha = current_alpha
    best_tot_e = current_tot_e
    best_parent = state.snapshot_parent()

    temp = 0.15 * current_guide if t_initial is None else float(t_initial)
    t_end = max(1e-9, float(t_min_ratio) * current_guide)
    if cooling_rate is None:
        cool = (t_end / max(temp, 1e-9)) ** (1.0 / max(1, max_iterations))
    else:
        cool = float(cooling_rate)

    convergence_history: list[dict[str, float | int]] = [
        {
            "iteration": 0,
            "current_obj": float(current_alpha),
            "best_obj": float(best_alpha),
            "temperature": float(temp),
        }
    ]

    accepted_moves = 0
    improving_moves = 0
    all_sensors = np.arange(1, n, dtype=np.int64)

    for it in range(1, max_iterations + 1):
        move_done = False
        used_swap = False
        moved_u = -1
        moved_v = -1
        old_pu = -1
        old_pv = -1

        focused = rng.random() < focus_prob
        if focused:
            rates = state.node_rate[1:]
            bn_candidates = all_sensors[rates >= 0.97 * current_alpha]
            bottleneck = int(rng.choice(bn_candidates)) if len(bn_candidates) > 0 else int(np.argmax(rates) + 1)
            sub_nodes = state.subtree_nodes(bottleneck)
            sub_set = set(sub_nodes)

            # Intento de intercambio 2-nodos (útil bajo K_max estricto)
            if len(sub_nodes) < n - 1 and rng.random() < swap_prob:
                outside_nodes = [i for i in range(1, n) if i not in sub_set]
                for _ in range(4):
                    u_cand = int(sub_nodes[int(rng.randint(len(sub_nodes)))])
                    v_cand = int(outside_nodes[int(rng.randint(len(outside_nodes)))])
                    if state.can_swap(u_cand, v_cand):
                        old_pu, old_pv = state.swap_parents(u_cand, v_cand)
                        moved_u, moved_v = u_cand, v_cand
                        used_swap = True
                        move_done = True
                        break

            # Si no hubo swap, reasignación enfocada en el subárbol del cuello de botella
            if not move_done:
                for _ in range(5):
                    if len(sub_nodes) > 1 and rng.random() < 0.70:
                        direct_kids = list(state.children[bottleneck])
                        if direct_kids and rng.random() < 0.65:
                            u_cand = int(direct_kids[int(rng.randint(len(direct_kids)))])
                        else:
                            desc = [x for x in sub_nodes if x != bottleneck]
                            u_cand = int(desc[int(rng.randint(len(desc)))])
                    else:
                        u_cand = bottleneck

                    cands = [
                        p
                        for p in state.neighbors_out[u_cand]
                        if state.can_reassign(u_cand, p)
                    ]
                    if not cands:
                        continue
                    if u_cand != bottleneck:
                        ext_cands = [p for p in cands if p not in sub_set]
                        if ext_cands and rng.random() < 0.85:
                            cands = ext_cands
                    p_new = int(cands[int(rng.randint(len(cands)))])
                    old_pu = state._move_unchecked(u_cand, p_new)
                    moved_u = u_cand
                    move_done = True
                    break

        if not move_done:
            # Movida exploratoria global sobre cualquier sensor
            for _ in range(6):
                u_cand = int(rng.randint(1, n))
                cands = [
                    p
                    for p in state.neighbors_out[u_cand]
                    if state.can_reassign(u_cand, p)
                ]
                if not cands:
                    continue
                p_new = int(cands[int(rng.randint(len(cands)))])
                old_pu = state._move_unchecked(u_cand, p_new)
                moved_u = u_cand
                move_done = True
                break

        improved_best = False
        if move_done:
            new_alpha = state.alpha()
            new_guide = state.guide_score()
            new_tot_e = state.total_energy()
            delta_guide = new_guide - current_guide

            if delta_guide <= 0.0 or rng.random() < math.exp(-delta_guide / max(temp, 1e-12)):
                accepted_moves += 1
                if new_alpha < current_alpha - 1e-9:
                    improving_moves += 1
                current_alpha = new_alpha
                current_guide = new_guide
                current_tot_e = new_tot_e

                if new_alpha < best_alpha - 1e-9 or (
                    abs(new_alpha - best_alpha) <= 1e-9 and new_tot_e < best_tot_e - 1e-6
                ):
                    best_alpha = new_alpha
                    best_tot_e = new_tot_e
                    best_parent = state.snapshot_parent()
                    improved_best = True
            else:
                if used_swap:
                    state.revert_swap(moved_u, moved_v, old_pu, old_pv)
                else:
                    state._move_unchecked(moved_u, old_pu)

        if it % 500 == 0:
            state.recompute_all()
            current_alpha = state.alpha()
            current_guide = state.guide_score()
            current_tot_e = state.total_energy()

        temp = max(t_end, temp * cool)

        if improved_best or it % sample_every == 0 or it == max_iterations:
            convergence_history.append(
                {
                    "iteration": int(it),
                    "current_obj": float(current_alpha),
                    "best_obj": float(best_alpha),
                    "temperature": float(temp),
                }
            )

    # Restaurar mejor solución encontrada y aplicar pulido local final
    state.restore_parent(best_parent)
    polish_moves = _local_polish(state, max_passes=30)
    improving_moves += polish_moves
    state.recompute_all()

    final_alpha = state.alpha()
    if final_alpha < best_alpha - 1e-9:
        best_alpha = final_alpha
        convergence_history.append(
            {
                "iteration": int(max_iterations + 1),
                "current_obj": float(final_alpha),
                "best_obj": float(best_alpha),
                "temperature": float(temp),
            }
        )

    topology, edges, flows = state.to_topology_and_flows()
    max_e = float(np.max(state.node_energy[1:]))
    tot_e = state.total_energy()
    lifetime = float(instance.nominal_battery_uJ) / final_alpha if final_alpha > 0.0 else float("inf")
    elapsed = time.perf_counter() - t0_wall

    return MetaheuristicResult(
        algorithm="simulated_annealing",
        mode=mode,
        topology=topology,
        edges=edges,
        flows=flows,
        objective=final_alpha,
        max_energy_uJ=max_e,
        total_energy_uJ=tot_e,
        lifetime_rounds=lifetime,
        solve_time_s=float(elapsed),
        iterations=int(max_iterations),
        accepted_moves=int(accepted_moves),
        improving_moves=int(improving_moves),
        seed=int(seed),
        convergence_history=convergence_history,
    )
