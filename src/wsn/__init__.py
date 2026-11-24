"""Arquitectura modular para enrutamiento energéticamente eficiente en WSN (Etapa 3)."""

from wsn.exact_solver import ExactSolveResult, solve_exact
from wsn.experiments import (
    run_all_experiments,
    run_exp1_verification,
    run_exp2_scalability,
    run_exp3_scenarios,
)
from wsn.instance import WSNInstance, generate_instance
from wsn.metaheuristic import MetaheuristicResult, solve_greedy, solve_simulated_annealing
from wsn.validator import validate_solution

__all__ = [
    "ExactSolveResult",
    "MetaheuristicResult",
    "WSNInstance",
    "generate_instance",
    "run_all_experiments",
    "run_exp1_verification",
    "run_exp2_scalability",
    "run_exp3_scenarios",
    "solve_exact",
    "solve_greedy",
    "solve_simulated_annealing",
    "validate_solution",
]
