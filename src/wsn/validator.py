"""Validador independiente del solver para auditar factibilidad, flujo y objetivo WSN."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import networkx as nx

from wsn.instance import WSNInstance


def _normalize_edges(
    topology: Mapping[int, int] | Sequence[tuple[int, int]],
    errors: list[str],
) -> list[tuple[int, int]]:
    """Convierte un diccionario {nodo: padre} o lista de aristas [(u, v)] en lista normalizada."""
    edges: list[tuple[int, int]] = []
    if isinstance(topology, Mapping):
        for u, v in topology.items():
            edges.append((int(u), int(v)))
        return edges

    for item in topology:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            errors.append(f"Elemento de arista inválido en topología: {item!r}")
            continue
        edges.append((int(item[0]), int(item[1])))
    return edges


def validate_solution(
    instance: WSNInstance,
    topology: Mapping[int, int] | Sequence[tuple[int, int]],
    mode: str = "extended",
    expected_objective: float | None = None,
    expected_flows: Mapping[tuple[int, int], float] | None = None,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Audita una topología WSN de forma totalmente independiente del solver.

    Verifica:
      1. Rango válido de nodos, ausencia de auto-lazos y sin aristas salientes del Gateway (0).
      2. Grado de salida exactamente igual a 1 para todo sensor 1..N-1.
      3. Distancias euclidianas <= D_max en toda arista activa.
      4. Límite de fan-in <= K_max en sensores (modo "extended" con K_max != None).
      5. Ausencia de ciclos (DAG) y conectividad total al Gateway (nodo 0).
      6. Recálculo independiente de flujos, consumos energéticos, tasa de agotamiento,
         vida útil estimada y concordancia opcional con flujos/objetivo reportados.
    """
    if mode not in ("base", "extended"):
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    errors: list[str] = []
    n = instance.N
    coords = instance.coords
    edges = _normalize_edges(topology, errors)

    seen_edges: set[tuple[int, int]] = set()
    out_edges: dict[int, list[int]] = defaultdict(list)
    in_edges: dict[int, list[int]] = defaultdict(list)

    for u, v in edges:
        if not (0 <= u < n) or not (0 <= v < n):
            errors.append(f"Arista ({u}, {v}) fuera de rango [0, {n - 1}].")
            continue
        if u == v:
            errors.append(f"Auto-lazo detectado en nodo {u}.")
            continue
        if u == 0:
            errors.append(f"El Gateway (nodo 0) no puede transmitir (arista {(u, v)}).")
            continue
        if (u, v) in seen_edges:
            errors.append(f"Arista duplicada {(u, v)}.")
            continue
        seen_edges.add((u, v))
        out_edges[u].append(v)
        in_edges[v].append(u)

        dist_uv = math.hypot(
            float(coords[u, 0]) - float(coords[v, 0]),
            float(coords[u, 1]) - float(coords[v, 1]),
        )
        if dist_uv > float(instance.D_max) + 1e-9:
            errors.append(
                f"Arista ({u}, {v}) excede D_max: d={dist_uv:.4f} m > {instance.D_max:.4f} m."
            )

    # Grado de salida = 1 para todo sensor 1..N-1
    for i in range(1, n):
        deg_out = len(out_edges[i])
        if deg_out != 1:
            errors.append(f"Sensor {i} tiene grado de salida {deg_out} (esperado 1).")

    # Fan-in <= K_max en modo extendido
    max_fan_in = max((len(in_edges[i]) for i in range(1, n)), default=0)
    if mode == "extended" and instance.K_max is not None:
        for i in range(1, n):
            fan_in_i = len(in_edges[i])
            if fan_in_i > int(instance.K_max):
                errors.append(
                    f"Sensor {i} excede fan-in máximo: {fan_in_i} > K_max={instance.K_max}."
                )

    # Grafo dirigido independiente para verificar ciclos y conectividad al Gateway
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v in seen_edges:
        graph.add_edge(u, v)

    is_dag = nx.is_directed_acyclic_graph(graph)
    if not is_dag:
        cycles = []
        for idx, cyc in enumerate(nx.simple_cycles(graph)):
            cycles.append(cyc)
            if idx >= 4:
                break
        errors.append(f"Se detectaron ciclos en la topología: {cycles}.")

    valid_out_degrees = all(len(out_edges[i]) == 1 for i in range(1, n)) and len(out_edges[0]) == 0
    hops: dict[int, int] = {}
    all_reach_gw = False

    if is_dag and valid_out_degrees:
        parent = {i: out_edges[i][0] for i in range(1, n)}
        all_reach_gw = True
        for i in range(1, n):
            curr = i
            steps = 0
            while curr != 0 and steps <= n:
                if curr not in parent:
                    break
                curr = parent[curr]
                steps += 1
            if curr != 0 or not nx.has_path(graph, i, 0):
                all_reach_gw = False
                errors.append(f"Sensor {i} no alcanza al Gateway (nodo 0).")
            else:
                hops[i] = steps
    elif is_dag:
        for i in range(1, n):
            if not nx.has_path(graph, i, 0):
                errors.append(f"Sensor {i} no tiene ruta hacia el Gateway (nodo 0).")

    # Recálculo independiente de flujos, consumos y vida útil si es un árbol enraizado válido
    recomputed_objective: float | None = None
    max_energy_uJ: float | None = None
    total_energy_uJ: float | None = None
    lifetime_rounds: float | None = None
    bottleneck_node: int | None = None
    node_flows_out: dict[int, float] = {}
    node_energies_uJ: dict[int, float] = {}
    node_depletion_rates: dict[int, float] = {}
    edge_flows: dict[tuple[int, int], float] = {}

    if is_dag and valid_out_degrees and all_reach_gw:
        parent = {i: out_edges[i][0] for i in range(1, n)}
        children: dict[int, list[int]] = defaultdict(list)
        for i, p in parent.items():
            children[p].append(i)

        topo_order = list(nx.topological_sort(graph))
        for u in topo_order:
            if u == 0:
                continue
            in_flow = sum(edge_flows[(c, u)] for c in children[u])
            out_flow = in_flow + float(instance.g[u])
            p = parent[u]
            edge_flows[(u, p)] = out_flow
            node_flows_out[u] = out_flow

        total_generated = float(sum(instance.g[1:]))
        gw_received = sum(edge_flows[(c, 0)] for c in children[0])
        if abs(gw_received - total_generated) > tol * max(1.0, total_generated):
            errors.append(
                f"Flujo recibido por Gateway ({gw_received:.6f}) difiere del generado ({total_generated:.6f})."
            )

        if expected_flows is not None:
            for edge, flow_val in edge_flows.items():
                exp_val = float(expected_flows.get(edge, 0.0))
                if abs(flow_val - exp_val) > tol * max(1.0, total_generated):
                    errors.append(
                        f"Flujo recalculado en {edge} ({flow_val:.4f}) difiere del esperado ({exp_val:.4f})."
                    )
            for edge, exp_val in expected_flows.items():
                if edge not in edge_flows and abs(float(exp_val)) > tol * max(1.0, total_generated):
                    errors.append(
                        f"Flujo inesperado {exp_val:.4f} en arista inactiva {edge}."
                    )

        for i in range(1, n):
            p = parent[i]
            in_flow = sum(edge_flows[(c, i)] for c in children[i])
            out_flow = node_flows_out[i]
            d_ip = math.hypot(
                float(coords[i, 0]) - float(coords[p, 0]),
                float(coords[i, 1]) - float(coords[p, 1]),
            )
            tx_cost_ip = (
                float(instance.e_tx_base)
                if mode == "base"
                else float(instance.e_elec) + float(instance.eps_amp) * (d_ip ** 2)
            )
            energy_i = float(instance.e_rx) * in_flow + tx_cost_ip * out_flow
            b_i = 1.0 if mode == "base" else float(instance.battery_levels[i])
            rate_i = energy_i / b_i

            node_energies_uJ[i] = energy_i
            node_depletion_rates[i] = rate_i

        bottleneck_node = max(node_depletion_rates, key=lambda idx: node_depletion_rates[idx])
        recomputed_objective = float(node_depletion_rates[bottleneck_node])
        max_energy_uJ = float(max(node_energies_uJ.values()))
        total_energy_uJ = float(sum(node_energies_uJ.values()))
        lifetime_rounds = (
            float(instance.nominal_battery_uJ) / recomputed_objective
            if recomputed_objective > 0.0
            else float("inf")
        )

        if expected_objective is not None:
            scale = max(1.0, abs(float(expected_objective)))
            if abs(recomputed_objective - float(expected_objective)) > tol * scale:
                errors.append(
                    f"Objetivo recalculado ({recomputed_objective:.6f}) difiere del esperado "
                    f"({float(expected_objective):.6f})."
                )

    return {
        "is_feasible": len(errors) == 0,
        "recomputed_objective": recomputed_objective,
        "max_energy_uJ": max_energy_uJ,
        "total_energy_uJ": total_energy_uJ,
        "lifetime_rounds": lifetime_rounds,
        "bottleneck_node": bottleneck_node,
        "max_fan_in": max_fan_in,
        "max_hops": max(hops.values()) if hops else 0,
        "node_flows_out": node_flows_out,
        "node_energies_uJ": node_energies_uJ,
        "node_depletion_rates": node_depletion_rates,
        "edge_flows": edge_flows,
        "errors": errors,
    }
