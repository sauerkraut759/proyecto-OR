"""Interfaz CLI limpia para la suite experimental y resolución individual WSN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from wsn.exact_solver import solve_exact
from wsn.experiments import (
    run_all_experiments,
    run_exp1_verification,
    run_exp2_scalability,
    run_exp3_scenarios,
)
from wsn.instance import WSNInstance, generate_instance
from wsn.metaheuristic import solve_greedy, solve_simulated_annealing
from wsn.plotting import plot_single_topology
from wsn.validator import validate_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proyecto-optimizacion",
        description="Enrutamiento energéticamente eficiente en WSN (Entrega Final ELI148).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta la suite completa (Exp 1, Exp 2 y Exp 3) y genera tablas/figuras.",
    )
    parser.add_argument("--exp1", action="store_true", help="Ejecuta solo Experimento 1.")
    parser.add_argument("--exp2", action="store_true", help="Ejecuta solo Experimento 2.")
    parser.add_argument("--exp3", action="store_true", help="Ejecuta solo Experimento 3.")
    parser.add_argument(
        "--solve-single",
        action="store_true",
        help="Resuelve y audita una única instancia parametrizada o cargada desde JSON.",
    )

    # Opciones para --solve-single
    parser.add_argument("--instance", type=str, default=None, help="Ruta a instancia JSON.")
    parser.add_argument("--nodes", type=int, default=15, help="Total de nodos N (incluye Gateway 0).")
    parser.add_argument("--area", type=float, default=70.0, help="Lado del área cuadrada (m).")
    parser.add_argument("--dmax", type=float, default=40.0, help="Radio máximo de cobertura D_max (m).")
    parser.add_argument("--kmax", type=int, default=5, help="Fan-in máximo por sensor (0 = sin límite).")
    parser.add_argument(
        "--gateway",
        type=float,
        nargs=2,
        default=None,
        metavar=("GX", "GY"),
        help="Coordenadas (X, Y) del Gateway.",
    )
    parser.add_argument(
        "--battery-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("B_MIN", "B_MAX"),
        help="Rango uniforme de baterías residuales B_i.",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "extended"],
        default="extended",
        help="Modo de formulación matemática.",
    )
    parser.add_argument(
        "--solver",
        choices=["milp", "greedy", "sa", "all"],
        default="all",
        help="Método de resolución para --solve-single.",
    )

    # Opciones generales
    parser.add_argument("--seed", type=int, default=42, help="Semilla determinista.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Límite de tiempo MILP (s).")
    parser.add_argument("--sa-runs", type=int, default=5, help="Corridas independientes de SA.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directorio de salida.")
    parser.add_argument("--no-plots", action="store_true", help="Omite la generación de figuras PNG.")
    parser.add_argument("--quiet", action="store_true", help="Minimiza la salida por consola.")
    return parser


def _run_single(args: argparse.Namespace) -> None:
    if args.instance:
        inst = WSNInstance.load_json(args.instance)
    else:
        k_val = None if args.kmax == 0 else args.kmax
        gw = tuple(args.gateway) if args.gateway is not None else None
        bat = tuple(args.battery_range) if args.battery_range is not None else None
        inst = generate_instance(
            N=args.nodes,
            area_size=args.area,
            D_max=args.dmax,
            gateway_pos=gw,
            seed=args.seed,
            K_max=k_val,
            battery_levels=bat,
        )

    out_dir = Path(args.output_dir)
    solvers = ["milp", "greedy", "sa"] if args.solver == "all" else [args.solver]

    for s_name in solvers:
        if s_name == "milp":
            res = solve_exact(inst, mode=args.mode, time_limit_s=args.time_limit)
            if res.objective is None:
                print(f"[MILP] estado={res.status} sin incumbente en {res.solve_time_s:.2f}s")
                continue
            audit = validate_solution(
                inst,
                res.topology,
                mode=args.mode,
                expected_objective=res.objective,
                expected_flows=res.flows,
            )
            print(
                f"[MILP] estado={res.status} | alpha={audit['recomputed_objective']:.4f} | "
                f"vida={audit['lifetime_rounds']:.1f} | t={res.solve_time_s:.4f}s | factible={audit['is_feasible']}"
            )
            if not args.no_plots:
                plot_single_topology(
                    inst,
                    audit,
                    f"Topología MILP ({args.mode}) — N={inst.N}",
                    out_dir / "figures" / f"single_milp_{args.mode}_N{inst.N}.png",
                )
        elif s_name == "greedy":
            res = solve_greedy(inst, mode=args.mode)
            audit = validate_solution(
                inst,
                res.topology,
                mode=args.mode,
                expected_objective=res.objective,
                expected_flows=res.flows,
            )
            print(
                f"[Greedy] alpha={audit['recomputed_objective']:.4f} | "
                f"vida={audit['lifetime_rounds']:.1f} | t={res.solve_time_s:.4f}s | factible={audit['is_feasible']}"
            )
        elif s_name == "sa":
            res = solve_simulated_annealing(inst, mode=args.mode, seed=args.seed)
            audit = validate_solution(
                inst,
                res.topology,
                mode=args.mode,
                expected_objective=res.objective,
                expected_flows=res.flows,
            )
            print(
                f"[SA] alpha={audit['recomputed_objective']:.4f} | "
                f"vida={audit['lifetime_rounds']:.1f} | t={res.solve_time_s:.4f}s | factible={audit['is_feasible']}"
            )
            if not args.no_plots:
                plot_single_topology(
                    inst,
                    audit,
                    f"Topología Simulated Annealing ({args.mode}) — N={inst.N}",
                    out_dir / "figures" / f"single_sa_{args.mode}_N{inst.N}.png",
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_plots = not args.no_plots
    verbose = not args.quiet

    if args.solve_single:
        _run_single(args)
        return 0

    run_specific = args.exp1 or args.exp2 or args.exp3
    if args.all or not run_specific:
        run_all_experiments(
            output_dir=args.output_dir,
            time_limit_s=args.time_limit,
            sa_runs=args.sa_runs,
            generate_plots=generate_plots,
            verbose=verbose,
        )
        return 0

    if args.exp1:
        res1 = run_exp1_verification(
            output_dir=args.output_dir,
            time_limit_s=args.time_limit,
            generate_plots=generate_plots,
        )
        if verbose:
            print(f"Exp 1 completado -> {res1['csv_path']}")
    if args.exp2:
        res2 = run_exp2_scalability(
            output_dir=args.output_dir,
            time_limit_s=args.time_limit,
            sa_runs=args.sa_runs,
            generate_plots=generate_plots,
        )
        if verbose:
            print(f"Exp 2 completado -> {res2['summary_csv']}")
    if args.exp3:
        res3 = run_exp3_scenarios(
            output_dir=args.output_dir,
            time_limit_s=args.time_limit,
            sa_runs=args.sa_runs,
            generate_plots=generate_plots,
        )
        if verbose:
            print(f"Exp 3 completado -> {res3['csv_path']}")
    return 0
