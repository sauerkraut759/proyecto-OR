"""Modelo MILP exacto en OR-Tools (SCIP) para enrutamiento energético en WSN."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from ortools.linear_solver import pywraplp

from wsn.instance import WSNInstance


_STATUS_MAP: dict[int, str] = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


@dataclass
class ExactSolveResult:
    """Resultado estructurado de la resolución exacta MILP (SCIP)."""

    mode: str
    status: str
    topology: dict[int, int]
    edges: list[tuple[int, int]]
    flows: dict[tuple[int, int], float]
    objective: float | None
    best_bound: float | None
    mip_gap: float | None
    solve_time_s: float
    num_nodes: int
    num_variables: int
    num_constraints: int


def solve_exact(
    instance: WSNInstance,
    mode: str = "extended",
    time_limit_s: float | None = 30.0,
    verbose: bool = False,
) -> ExactSolveResult:
    """Resuelve el problema de topología y enrutamiento WSN mediante MILP (SCIP).

    Modos soportados:
      - "base": formulación del Avance 2 (energía TX constante `e_tx_base`, sin límite
        de fan-in y baterías homogéneas B_i = 1.0).
      - "extended": formulación de la Etapa 3 (energía TX dependiente de distancia
        e_elec + eps_amp * d_ij^2, límite máximo de fan-in K_max por sensor y
        minimización de tasa máxima de agotamiento alpha = max_i E_i / B_i).
    """
    if mode not in ("base", "extended"):
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("El solver SCIP de OR-Tools no está disponible en este entorno.")

    if time_limit_s is not None:
        solver.SetTimeLimit(int(round(float(time_limit_s) * 1000.0)))
    if verbose:
        solver.EnableOutput()
    # Limitar rondas de separación en el nodo raíz evita estancamiento en planos de corte
    # cuando los coeficientes energéticos dependientes de distancia son continuos.
    solver.SetSolverSpecificParametersAsString("separating/maxroundsroot = 5")

    n = instance.N
    edges = instance.feasible_edges()
    tx_cost = instance.tx_cost_matrix(mode=mode)
    big_m = float(sum(instance.g[1:]))

    in_edges: dict[int, list[int]] = {i: [] for i in range(n)}
    out_edges: dict[int, list[int]] = {i: [] for i in range(1, n)}
    for i, j in edges:
        out_edges[i].append(j)
        in_edges[j].append(i)

    # Variables de decisión
    x: dict[tuple[int, int], pywraplp.Variable] = {}
    f: dict[tuple[int, int], pywraplp.Variable] = {}
    for i, j in edges:
        x[i, j] = solver.BoolVar(f"x_{i}_{j}")
        f[i, j] = solver.NumVar(0.0, big_m, f"f_{i}_{j}")

    alpha = solver.NumVar(0.0, solver.infinity(), "alpha")

    # 1. Asignación única por sensor (árbol dirigido hacia nodo 0)
    for i in range(1, n):
        solver.Add(
            solver.Sum([x[i, j] for j in out_edges[i]]) == 1,
            f"assign_{i}",
        )

    # 2. Conservación de flujo
    for i in range(1, n):
        inflow = solver.Sum([f[k, i] for k in in_edges[i]])
        outflow = solver.Sum([f[i, j] for j in out_edges[i]])
        solver.Add(
            inflow + float(instance.g[i]) == outflow,
            f"flow_balance_{i}",
        )

    # 3. Acoplamiento Big-M ajustado entre topología y flujo
    for i, j in edges:
        m_ij = big_m if j == 0 else max(float(instance.g[i]), big_m - float(instance.g[j]))
        solver.Add(f[i, j] <= m_ij * x[i, j], f"coupling_ub_{i}_{j}")
        solver.Add(f[i, j] >= float(instance.g[i]) * x[i, j], f"coupling_lb_{i}_{j}")

    # 4. Cota Min-Max de consumo energético / tasa de agotamiento
    for i in range(1, n):
        b_i = 1.0 if mode == "base" else float(instance.battery_levels[i])
        rx_energy = float(instance.e_rx) * solver.Sum([f[k, i] for k in in_edges[i]])
        tx_energy = solver.Sum([float(tx_cost[i, j]) * f[i, j] for j in out_edges[i]])
        solver.Add(rx_energy + tx_energy <= b_i * alpha, f"minmax_{i}")

    # 5. Límite de fan-in máximo por sensor repetidor (solo modo extendido)
    if mode == "extended" and instance.K_max is not None:
        for i in range(1, n):
            solver.Add(
                solver.Sum([x[k, i] for k in in_edges[i]]) <= int(instance.K_max),
                f"fan_in_{i}",
            )

    solver.Minimize(alpha)

    t0 = time.perf_counter()
    status_code = solver.Solve()
    solve_time_s = time.perf_counter() - t0

    status_str = _STATUS_MAP.get(status_code, f"UNKNOWN_{status_code}")
    num_bb_nodes = int(solver.nodes())
    num_vars = int(solver.NumVariables())
    num_cons = int(solver.NumConstraints())

    best_bound: float | None = None
    try:
        raw_bound = float(solver.Objective().BestBound())
        if math.isfinite(raw_bound) and abs(raw_bound) < 1e18:
            best_bound = 0.0 if -1e-9 <= raw_bound < 0.0 else raw_bound
    except Exception:
        best_bound = None

    if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        topology: dict[int, int] = {}
        active_edges: list[tuple[int, int]] = []
        active_flows: dict[tuple[int, int], float] = {}
        for i, j in edges:
            if x[i, j].solution_value() > 0.5:
                topology[i] = j
                active_edges.append((i, j))
                active_flows[(i, j)] = float(f[i, j].solution_value())

        active_edges.sort()
        objective = float(alpha.solution_value())
        if status_code == pywraplp.Solver.OPTIMAL:
            mip_gap: float | None = 0.0
            if best_bound is None:
                best_bound = objective
        elif best_bound is not None:
            mip_gap = max(0.0, (objective - best_bound) / max(1e-12, abs(objective)))
        else:
            mip_gap = None
    else:
        topology = {}
        active_edges = []
        active_flows = {}
        objective = None
        mip_gap = None

    return ExactSolveResult(
        mode=mode,
        status=status_str,
        topology=topology,
        edges=active_edges,
        flows=active_flows,
        objective=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        solve_time_s=float(solve_time_s),
        num_nodes=num_bb_nodes,
        num_variables=num_vars,
        num_constraints=num_cons,
    )
