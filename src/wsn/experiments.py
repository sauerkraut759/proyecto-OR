"""Suite experimental reproducible (Exp 1, Exp 2 y Exp 3) para la Entrega Final."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from wsn.exact_solver import solve_exact
from wsn.instance import WSNInstance, generate_instance
from wsn.metaheuristic import solve_greedy, solve_simulated_annealing
from wsn.plotting import (
    plot_objective_and_gap,
    plot_operational_scenarios,
    plot_sa_convergence,
    plot_scalability_time,
    plot_scenarios_metrics,
    plot_topology_comparison,
)
from wsn.validator import validate_solution


SCALABILITY_CONFIGS: list[tuple[int, float]] = [
    (15, 70.0),
    (25, 90.0),
    (40, 115.0),
    (60, 140.0),
    (100, 180.0),
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt_num(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}"


def _audit_or_raise(
    instance: WSNInstance,
    topology: dict[int, int],
    mode: str,
    expected_objective: float,
    expected_flows: dict[tuple[int, int], float],
    label: str,
) -> dict[str, Any]:
    audit = validate_solution(
        instance,
        topology,
        mode=mode,
        expected_objective=expected_objective,
        expected_flows=expected_flows,
    )
    if not audit["is_feasible"]:
        raise RuntimeError(
            f"Validación independiente fallida en {label}: {audit['errors']}"
        )
    return audit


def run_exp1_verification(
    output_dir: str | Path = "results",
    time_limit_s: float = 30.0,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Exp 1: Verificación N=15, seed=42 comparando modelo Base vs Extendido."""
    out_root = Path(output_dir)
    inst = generate_instance(
        N=15,
        area_size=70.0,
        D_max=40.0,
        gateway_pos=(35.0, 35.0),
        seed=42,
        g=400.0,
        e_elec=0.050,
        eps_amp=1.0e-4,
        e_rx=0.0828,
        e_tx_base=0.1104,
        K_max=5,
        battery_levels=None,
        name="exp1_n15_seed42",
    )
    json_path = inst.save_json(out_root / "instances" / "exp1_n15_seed42.json")
    inst = WSNInstance.load_json(json_path)

    milp_base = solve_exact(inst, mode="base", time_limit_s=time_limit_s)
    audit_base = _audit_or_raise(
        inst,
        milp_base.topology,
        mode="base",
        expected_objective=float(milp_base.objective),
        expected_flows=milp_base.flows,
        label="Exp1 MILP Base",
    )

    milp_ext = solve_exact(inst, mode="extended", time_limit_s=time_limit_s)
    audit_ext = _audit_or_raise(
        inst,
        milp_ext.topology,
        mode="extended",
        expected_objective=float(milp_ext.objective),
        expected_flows=milp_ext.flows,
        label="Exp1 MILP Extendido",
    )

    greedy_ext = solve_greedy(inst, mode="extended")
    audit_greedy = _audit_or_raise(
        inst,
        greedy_ext.topology,
        mode="extended",
        expected_objective=greedy_ext.objective,
        expected_flows=greedy_ext.flows,
        label="Exp1 Greedy Extendido",
    )

    sa_ext = solve_simulated_annealing(inst, mode="extended", seed=42)
    audit_sa = _audit_or_raise(
        inst,
        sa_ext.topology,
        mode="extended",
        expected_objective=sa_ext.objective,
        expected_flows=sa_ext.flows,
        label="Exp1 SA Extendido",
    )

    rows = [
        {
            "metodo": "MILP_SCIP",
            "modo": "base",
            "estado": milp_base.status,
            "objetivo_alpha": round(audit_base["recomputed_objective"], 6),
            "cota_inferior": _fmt_num(milp_base.best_bound, 6),
            "mip_gap_pct": _fmt_num(100.0 * milp_base.mip_gap if milp_base.mip_gap is not None else None, 4),
            "max_energia_uJ": round(audit_base["max_energy_uJ"], 6),
            "total_energia_uJ": round(audit_base["total_energy_uJ"], 6),
            "vida_util_ciclos": round(audit_base["lifetime_rounds"], 2),
            "nodo_cuello_botella": audit_base["bottleneck_node"],
            "fan_in_max": audit_base["max_fan_in"],
            "saltos_max": audit_base["max_hops"],
            "tiempo_s": round(milp_base.solve_time_s, 4),
            "nodos_bb": milp_base.num_nodes,
            "es_factible": audit_base["is_feasible"],
            "errores": len(audit_base["errors"]),
        },
        {
            "metodo": "MILP_SCIP",
            "modo": "extended",
            "estado": milp_ext.status,
            "objetivo_alpha": round(audit_ext["recomputed_objective"], 6),
            "cota_inferior": _fmt_num(milp_ext.best_bound, 6),
            "mip_gap_pct": _fmt_num(100.0 * milp_ext.mip_gap if milp_ext.mip_gap is not None else None, 4),
            "max_energia_uJ": round(audit_ext["max_energy_uJ"], 6),
            "total_energia_uJ": round(audit_ext["total_energy_uJ"], 6),
            "vida_util_ciclos": round(audit_ext["lifetime_rounds"], 2),
            "nodo_cuello_botella": audit_ext["bottleneck_node"],
            "fan_in_max": audit_ext["max_fan_in"],
            "saltos_max": audit_ext["max_hops"],
            "tiempo_s": round(milp_ext.solve_time_s, 4),
            "nodos_bb": milp_ext.num_nodes,
            "es_factible": audit_ext["is_feasible"],
            "errores": len(audit_ext["errors"]),
        },
        {
            "metodo": "Greedy_Balanceado",
            "modo": "extended",
            "estado": "HEURISTIC",
            "objetivo_alpha": round(audit_greedy["recomputed_objective"], 6),
            "cota_inferior": "",
            "mip_gap_pct": round(100.0 * (audit_greedy["recomputed_objective"] - audit_ext["recomputed_objective"]) / audit_ext["recomputed_objective"], 4),
            "max_energia_uJ": round(audit_greedy["max_energy_uJ"], 6),
            "total_energia_uJ": round(audit_greedy["total_energy_uJ"], 6),
            "vida_util_ciclos": round(audit_greedy["lifetime_rounds"], 2),
            "nodo_cuello_botella": audit_greedy["bottleneck_node"],
            "fan_in_max": audit_greedy["max_fan_in"],
            "saltos_max": audit_greedy["max_hops"],
            "tiempo_s": round(greedy_ext.solve_time_s, 6),
            "nodos_bb": 0,
            "es_factible": audit_greedy["is_feasible"],
            "errores": len(audit_greedy["errors"]),
        },
        {
            "metodo": "Simulated_Annealing",
            "modo": "extended",
            "estado": "METAHEURISTIC",
            "objetivo_alpha": round(audit_sa["recomputed_objective"], 6),
            "cota_inferior": "",
            "mip_gap_pct": round(100.0 * (audit_sa["recomputed_objective"] - audit_ext["recomputed_objective"]) / audit_ext["recomputed_objective"], 4),
            "max_energia_uJ": round(audit_sa["max_energy_uJ"], 6),
            "total_energia_uJ": round(audit_sa["total_energy_uJ"], 6),
            "vida_util_ciclos": round(audit_sa["lifetime_rounds"], 2),
            "nodo_cuello_botella": audit_sa["bottleneck_node"],
            "fan_in_max": audit_sa["max_fan_in"],
            "saltos_max": audit_sa["max_hops"],
            "tiempo_s": round(sa_ext.solve_time_s, 4),
            "nodos_bb": 0,
            "es_factible": audit_sa["is_feasible"],
            "errores": len(audit_sa["errors"]),
        },
    ]

    csv_path = _write_csv(out_root / "tables" / "exp1_base_vs_extended.csv", rows)
    fig_path = None
    if generate_plots:
        fig_path = plot_topology_comparison(
            inst,
            audit_base,
            audit_ext,
            out_root / "figures" / "exp1_topologia_base_vs_extendido.png",
        )

    return {
        "rows": rows,
        "csv_path": csv_path,
        "fig_path": fig_path,
        "instance_path": json_path,
    }


def run_exp2_scalability(
    output_dir: str | Path = "results",
    time_limit_s: float = 30.0,
    sa_runs: int = 5,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Exp 2: Escalabilidad N in [15, 25, 40, 60, 100] (MILP vs Greedy vs 5x SA)."""
    out_root = Path(output_dir)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    convergence_panels: list[dict[str, Any]] = []

    for n_nodes, area in SCALABILITY_CONFIGS:
        inst = generate_instance(
            N=n_nodes,
            area_size=area,
            D_max=40.0,
            gateway_pos=(area / 2.0, area / 2.0),
            seed=42,
            g=400.0,
            e_elec=0.050,
            eps_amp=1.0e-4,
            e_rx=0.0828,
            e_tx_base=0.1104,
            K_max=5,
            battery_levels=None,
            name=f"exp2_n{n_nodes}_seed42",
        )
        inst.save_json(out_root / "instances" / f"exp2_n{n_nodes}_seed42.json")

        # 1. Solver exacto MILP (SCIP)
        milp = solve_exact(inst, mode="extended", time_limit_s=time_limit_s)
        milp_feasible: bool | None = None
        if milp.status in ("OPTIMAL", "FEASIBLE") and milp.objective is not None:
            milp_audit = _audit_or_raise(
                inst,
                milp.topology,
                mode="extended",
                expected_objective=milp.objective,
                expected_flows=milp.flows,
                label=f"Exp2 MILP N={n_nodes}",
            )
            milp_feasible = bool(milp_audit["is_feasible"])

        # 2. Heurística Greedy
        greedy = solve_greedy(inst, mode="extended")
        greedy_audit = _audit_or_raise(
            inst,
            greedy.topology,
            mode="extended",
            expected_objective=greedy.objective,
            expected_flows=greedy.flows,
            label=f"Exp2 Greedy N={n_nodes}",
        )

        # 3. Simulated Annealing (5 corridas)
        sa_results = []
        sa_audits = []
        for run_idx in range(sa_runs):
            sa_seed = 101 + run_idx
            res = solve_simulated_annealing(inst, mode="extended", seed=sa_seed)
            aud = _audit_or_raise(
                inst,
                res.topology,
                mode="extended",
                expected_objective=res.objective,
                expected_flows=res.flows,
                label=f"Exp2 SA N={n_nodes} run={run_idx + 1}",
            )
            sa_results.append(res)
            sa_audits.append(aud)

        # Referencia para cálculo de Gap %
        if milp.status == "OPTIMAL" and milp.objective is not None:
            ref_val = float(milp.objective)
            ref_type = "OPTIMO_MILP"
        elif milp.best_bound is not None and milp.best_bound > 0.0:
            ref_val = float(milp.best_bound)
            ref_type = "COTA_DUAL_SCIP"
        else:
            ref_val = None
            ref_type = "SIN_COTA"

        sa_objs = np.array([r.objective for r in sa_results], dtype=np.float64)
        sa_times = np.array([r.solve_time_s for r in sa_results], dtype=np.float64)
        best_idx = int(np.argmin(sa_objs))
        best_sa = sa_results[best_idx]
        best_sa_audit = sa_audits[best_idx]

        for run_idx, (res, aud) in enumerate(zip(sa_results, sa_audits)):
            gap_ref = (100.0 * (res.objective - ref_val) / ref_val) if ref_val else None
            detail_rows.append(
                {
                    "N": n_nodes,
                    "area_m": area,
                    "corrida": run_idx + 1,
                    "sa_seed": res.seed,
                    "objetivo_alpha": round(res.objective, 6),
                    "gap_vs_ref_pct": round(gap_ref, 4) if gap_ref is not None else "",
                    "vida_util_ciclos": round(aud["lifetime_rounds"], 2),
                    "total_energia_uJ": round(aud["total_energy_uJ"], 4),
                    "tiempo_s": round(res.solve_time_s, 4),
                    "iteraciones": res.iterations,
                    "movidas_aceptadas": res.accepted_moves,
                    "es_factible": aud["is_feasible"],
                }
            )

        sa_best_val = float(np.min(sa_objs))
        sa_mean_val = float(np.mean(sa_objs))
        sa_std_val = float(np.std(sa_objs, ddof=1)) if sa_runs > 1 else 0.0
        sa_mean_t = float(np.mean(sa_times))
        sa_std_t = float(np.std(sa_times, ddof=1)) if sa_runs > 1 else 0.0

        greedy_gap = (100.0 * (greedy.objective - ref_val) / ref_val) if ref_val else None
        sa_best_gap = (100.0 * (sa_best_val - ref_val) / ref_val) if ref_val else None
        sa_mean_gap = (100.0 * (sa_mean_val - ref_val) / ref_val) if ref_val else None
        sa_vs_incumbent = (
            100.0 * (sa_best_val - milp.objective) / milp.objective
            if milp.objective is not None and milp.objective > 0
            else None
        )

        summary_rows.append(
            {
                "N": n_nodes,
                "area_m": area,
                "aristas_factibles": len(inst.feasible_edges()),
                "milp_estado": milp.status,
                "milp_objetivo": round(milp.objective, 6) if milp.objective is not None else None,
                "milp_cota_inferior": round(milp.best_bound, 6) if milp.best_bound is not None else None,
                "milp_gap_pct": round(100.0 * milp.mip_gap, 4) if milp.mip_gap is not None else None,
                "milp_tiempo_s": round(milp.solve_time_s, 4),
                "milp_nodos_bb": milp.num_nodes,
                "milp_factible": milp_feasible,
                "greedy_objetivo": round(greedy.objective, 6),
                "greedy_tiempo_s": round(greedy.solve_time_s, 6),
                "greedy_gap_vs_ref_pct": round(greedy_gap, 4) if greedy_gap is not None else None,
                "greedy_factible": greedy_audit["is_feasible"],
                "sa_mejor_objetivo": round(sa_best_val, 6),
                "sa_promedio_objetivo": round(sa_mean_val, 6),
                "sa_std_objetivo": round(sa_std_val, 6),
                "sa_vida_util_mejor": round(best_sa_audit["lifetime_rounds"], 2),
                "sa_tiempo_promedio_s": round(sa_mean_t, 4),
                "sa_tiempo_std_s": round(sa_std_t, 4),
                "sa_mejor_gap_vs_ref_pct": round(sa_best_gap, 4) if sa_best_gap is not None else None,
                "sa_promedio_gap_vs_ref_pct": round(sa_mean_gap, 4) if sa_mean_gap is not None else None,
                "sa_vs_milp_incumbente_pct": round(sa_vs_incumbent, 4) if sa_vs_incumbent is not None else None,
                "sa_corridas_factibles": f"{sum(1 for a in sa_audits if a['is_feasible'])}/{sa_runs}",
                "referencia_tipo": ref_type,
                "referencia_valor": round(ref_val, 6) if ref_val is not None else None,
            }
        )

        if n_nodes in (25, 100):
            convergence_panels.append(
                {
                    "N": n_nodes,
                    "title": f"Red N={n_nodes} ({area:.0f}×{area:.0f} m, semilla={best_sa.seed})",
                    "history": best_sa.convergence_history,
                    "greedy_obj": greedy.objective,
                    "milp_obj": milp.objective,
                    "milp_bound": milp.best_bound,
                    "milp_status": milp.status,
                }
            )

    summary_csv = _write_csv(out_root / "tables" / "exp2_scalability_summary.csv", summary_rows)
    detail_csv = _write_csv(out_root / "tables" / "exp2_sa_runs_detail.csv", detail_rows)

    fig_paths: list[Path] = []
    if generate_plots:
        fig_paths.append(
            plot_scalability_time(
                summary_rows,
                out_root / "figures" / "exp2_escalabilidad_tiempo.png",
                time_limit_s=time_limit_s,
            )
        )
        fig_paths.append(
            plot_objective_and_gap(
                summary_rows,
                out_root / "figures" / "exp2_objetivo_y_gap.png",
            )
        )
        fig_paths.append(
            plot_sa_convergence(
                convergence_panels,
                out_root / "figures" / "exp2_convergencia_sa.png",
            )
        )

    return {
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
        "fig_paths": fig_paths,
    }


def _find_shared_scenario_layout(
    n_nodes: int = 25,
    area_size: float = 90.0,
    d_max: float = 40.0,
    start_seed: int = 42,
) -> tuple[WSNInstance, WSNInstance, WSNInstance]:
    """Construye los 3 escenarios sobre exactamente las mismas coordenadas de sensores."""
    for cand_seed in range(start_seed, start_seed + 500):
        rng = np.random.RandomState(cand_seed)
        coords = rng.rand(n_nodes, 2) * area_size

        inst_a = WSNInstance(
            N=n_nodes,
            area_size=area_size,
            D_max=d_max,
            gateway_pos=(area_size / 2.0, area_size / 2.0),
            seed=cand_seed,
            coords=coords,
            g=400.0,
            e_elec=0.050,
            eps_amp=1.0e-4,
            e_rx=0.0828,
            e_tx_base=0.1104,
            K_max=5,
            battery_levels=None,
            name="exp3_escenario_A_gw_centrado",
        )
        inst_b = inst_a.with_overrides(
            gateway_pos=(5.0, 5.0),
            K_max=5,
            battery_levels=None,
            name="exp3_escenario_B_gw_esquina",
        )
        inst_c = inst_a.with_overrides(
            gateway_pos=(area_size / 2.0, area_size / 2.0),
            K_max=3,
            battery_levels=(0.6, 1.0),
            seed=cand_seed,
            name="exp3_escenario_C_k3_bat_het",
        )

        if (
            inst_a.has_kmax_spanning_tree(5)
            and inst_b.has_kmax_spanning_tree(5)
            and inst_c.has_kmax_spanning_tree(3)
        ):
            return inst_a, inst_b, inst_c

    raise RuntimeError("No se encontró un layout compartido factible para los 3 escenarios.")


def run_exp3_scenarios(
    output_dir: str | Path = "results",
    time_limit_s: float = 30.0,
    sa_runs: int = 5,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Exp 3: 3 Escenarios Operacionales en N=25 (GW centrado, GW esquina (5,5), K_max=3 + B_i)."""
    out_root = Path(output_dir)
    inst_a, inst_b, inst_c = _find_shared_scenario_layout(
        n_nodes=25, area_size=90.0, d_max=40.0, start_seed=42
    )

    scenario_specs = [
        (
            "A",
            "Escenario A: GW Centrado (45,45)",
            "A: GW Centrado\n(K_max=5, B=1.0)",
            "Escenario A: Gateway Centrado (45, 45)\nK_max = 5 | Baterías homogéneas B_i = 1.0",
            inst_a,
            False,
        ),
        (
            "B",
            "Escenario B: GW Esquina (5,5)",
            "B: GW Esquina\n(5, 5) | K_max=5",
            "Escenario B: Gateway en Esquina (5, 5)\nK_max = 5 | Baterías homogéneas B_i = 1.0",
            inst_b,
            False,
        ),
        (
            "C",
            "Escenario C: K_max=3 + Baterías [0.6, 1.0]",
            "C: K_max=3\nB_i ∈ [0.6, 1.0]",
            "Escenario C: K_max = 3 + Baterías Heterogéneas\nGW Centrado (45, 45) | B_i ∈ [0.6, 1.0]",
            inst_c,
            True,
        ),
    ]

    scenario_rows: list[dict[str, Any]] = []
    plot_panels: list[dict[str, Any]] = []

    for code, desc, short_lbl, plot_title, inst, ann_bat in scenario_specs:
        inst.save_json(out_root / "instances" / f"exp3_scenario_{code}.json")

        milp = solve_exact(inst, mode="extended", time_limit_s=time_limit_s)
        if milp.objective is None:
            raise RuntimeError(f"MILP no encontró solución factible en Escenario {code}.")
        milp_audit = _audit_or_raise(
            inst,
            milp.topology,
            mode="extended",
            expected_objective=milp.objective,
            expected_flows=milp.flows,
            label=f"Exp3 Escenario {code} MILP",
        )

        greedy = solve_greedy(inst, mode="extended")
        greedy_audit = _audit_or_raise(
            inst,
            greedy.topology,
            mode="extended",
            expected_objective=greedy.objective,
            expected_flows=greedy.flows,
            label=f"Exp3 Escenario {code} Greedy",
        )

        sa_results = []
        sa_audits = []
        for run_idx in range(sa_runs):
            res = solve_simulated_annealing(inst, mode="extended", seed=101 + run_idx)
            aud = _audit_or_raise(
                inst,
                res.topology,
                mode="extended",
                expected_objective=res.objective,
                expected_flows=res.flows,
                label=f"Exp3 Escenario {code} SA run={run_idx + 1}",
            )
            sa_results.append(res)
            sa_audits.append(aud)

        sa_objs = np.array([r.objective for r in sa_results], dtype=np.float64)
        best_idx = int(np.argmin(sa_objs))
        best_sa_audit = sa_audits[best_idx]

        ref = float(milp.objective) if milp.status == "OPTIMAL" else float(milp.best_bound or milp.objective)
        sa_best = float(np.min(sa_objs))
        sa_mean = float(np.mean(sa_objs))
        sa_std = float(np.std(sa_objs, ddof=1)) if sa_runs > 1 else 0.0

        scenario_rows.append(
            {
                "escenario": code,
                "descripcion": desc,
                "etiqueta_corta": short_lbl,
                "seed_layout": inst.seed,
                "gateway_x": inst.gateway_pos[0],
                "gateway_y": inst.gateway_pos[1],
                "K_max": inst.K_max,
                "bat_min": round(float(np.min(inst.battery_levels[1:])), 4),
                "bat_max": round(float(np.max(inst.battery_levels[1:])), 4),
                "milp_estado": milp.status,
                "milp_objetivo": round(milp_audit["recomputed_objective"], 6),
                "milp_max_energia_uJ": round(milp_audit["max_energy_uJ"], 6),
                "milp_vida_util_ciclos": round(milp_audit["lifetime_rounds"], 2),
                "milp_cuello_botella": milp_audit["bottleneck_node"],
                "milp_fan_in_max": milp_audit["max_fan_in"],
                "milp_saltos_max": milp_audit["max_hops"],
                "milp_tiempo_s": round(milp.solve_time_s, 4),
                "milp_factible": milp_audit["is_feasible"],
                "greedy_objetivo": round(greedy_audit["recomputed_objective"], 6),
                "greedy_vida_util_ciclos": round(greedy_audit["lifetime_rounds"], 2),
                "greedy_gap_pct": round(100.0 * (greedy_audit["recomputed_objective"] - ref) / ref, 4),
                "greedy_factible": greedy_audit["is_feasible"],
                "sa_mejor_objetivo": round(sa_best, 6),
                "sa_promedio_objetivo": round(sa_mean, 6),
                "sa_std_objetivo": round(sa_std, 6),
                "sa_vida_util_ciclos": round(best_sa_audit["lifetime_rounds"], 2),
                "sa_mejor_gap_pct": round(100.0 * (sa_best - ref) / ref, 4),
                "sa_tiempo_promedio_s": round(float(np.mean([r.solve_time_s for r in sa_results])), 4),
                "sa_corridas_factibles": f"{sum(1 for a in sa_audits if a['is_feasible'])}/{sa_runs}",
            }
        )

        plot_panels.append(
            {
                "code": code,
                "title": plot_title,
                "instance": inst,
                "audit": milp_audit,
                "annotate_batteries": ann_bat,
            }
        )

    csv_path = _write_csv(out_root / "tables" / "exp3_operational_scenarios.csv", scenario_rows)
    fig_paths: list[Path] = []
    if generate_plots:
        fig_paths.append(
            plot_operational_scenarios(
                plot_panels,
                out_root / "figures" / "exp3_escenarios_operacionales.png",
            )
        )
        fig_paths.append(
            plot_scenarios_metrics(
                scenario_rows,
                out_root / "figures" / "exp3_comparacion_escenarios.png",
            )
        )

    return {
        "rows": scenario_rows,
        "csv_path": csv_path,
        "fig_paths": fig_paths,
    }


def run_all_experiments(
    output_dir: str | Path = "results",
    time_limit_s: float = 30.0,
    sa_runs: int = 5,
    generate_plots: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta de forma reproducible Exp 1, Exp 2 y Exp 3 y guarda tablas y gráficos."""
    if verbose:
        print("=== Ejecutando Experimento 1: Verificación N=15 (Base vs Extendido) ===")
    exp1 = run_exp1_verification(
        output_dir=output_dir,
        time_limit_s=time_limit_s,
        generate_plots=generate_plots,
    )

    if verbose:
        print("=== Ejecutando Experimento 2: Escalabilidad N=[15, 25, 40, 60, 100] ===")
    exp2 = run_exp2_scalability(
        output_dir=output_dir,
        time_limit_s=time_limit_s,
        sa_runs=sa_runs,
        generate_plots=generate_plots,
    )

    if verbose:
        print("=== Ejecutando Experimento 3: Escenarios Operacionales A, B y C (N=25) ===")
    exp3 = run_exp3_scenarios(
        output_dir=output_dir,
        time_limit_s=time_limit_s,
        sa_runs=sa_runs,
        generate_plots=generate_plots,
    )

    if verbose:
        print("\n--- Resumen Exp 1 (N=15, seed=42) ---")
        for r in exp1["rows"]:
            print(
                f"  {r['metodo']:20s} | modo={r['modo']:8s} | alpha={r['objetivo_alpha']:8.4f} | "
                f"E_max={r['max_energia_uJ']:8.4f} | vida={r['vida_util_ciclos']:9.1f} | "
                f"t={r['tiempo_s']:.4f}s | factible={r['es_factible']}"
            )
        print("\n--- Resumen Exp 2 (Escalabilidad) ---")
        for r in exp2["summary_rows"]:
            milp_o = f"{r['milp_objetivo']:.2f}" if r["milp_objetivo"] is not None else "S/S"
            milp_bb = f"{r['milp_cota_inferior']:.2f}" if r["milp_cota_inferior"] is not None else "N/A"
            sa_g = f"{r['sa_mejor_gap_vs_ref_pct']:.2f}%" if r["sa_mejor_gap_vs_ref_pct"] is not None else "N/A"
            print(
                f"  N={r['N']:3d} | MILP={milp_o:>7s} ({r['milp_estado']:8s}, BB={milp_bb:>6s}, {r['milp_tiempo_s']:6.2f}s) | "
                f"Greedy={r['greedy_objetivo']:7.2f} | SA_best={r['sa_mejor_objetivo']:7.2f} "
                f"(mean={r['sa_promedio_objetivo']:7.2f}±{r['sa_std_objetivo']:5.2f}, gap={sa_g}, {r['sa_tiempo_promedio_s']:.3f}s) | "
                f"SA_ok={r['sa_corridas_factibles']}"
            )
        print("\n--- Resumen Exp 3 (Escenarios N=25) ---")
        for r in exp3["rows"]:
            print(
                f"  Esc {r['escenario']} ({r['descripcion']:38s}) | MILP={r['milp_objetivo']:7.2f} ({r['milp_estado']}) | "
                f"Greedy={r['greedy_objetivo']:7.2f} | SA_best={r['sa_mejor_objetivo']:7.2f} | "
                f"vida_MILP={r['milp_vida_util_ciclos']:8.1f} | SA_ok={r['sa_corridas_factibles']}"
            )

    return {"exp1": exp1, "exp2": exp2, "exp3": exp3}
