"""Visualizaciones de alta resolución (300 DPI) en español para análisis y presentación WSN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np

from wsn.instance import WSNInstance


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _draw_topology_panel(
    ax: plt.Axes,
    instance: WSNInstance,
    audit: dict[str, Any],
    title: str,
    norm: Normalize,
    cmap: Any,
    annotate_batteries: bool = False,
) -> None:
    """Dibuja una topología de árbol dirigida sobre un eje Matplotlib."""
    coords = instance.coords
    n = instance.N
    edge_flows: dict[tuple[int, int], float] = audit["edge_flows"]
    rates: dict[int, float] = audit["node_depletion_rates"]
    bottleneck = audit["bottleneck_node"]
    max_flow = max(edge_flows.values()) if edge_flows else 1.0

    # Radio de cobertura sutil desde el Gateway
    gw_circle = plt.Circle(
        (coords[0, 0], coords[0, 1]),
        instance.D_max,
        color="#94a3b8",
        fill=False,
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
        zorder=1,
    )
    ax.add_patch(gw_circle)

    # Aristas dirigidas ponderadas por flujo
    for (u, v), flow in edge_flows.items():
        rel = flow / max(max_flow, 1e-9)
        lw = 1.0 + 2.8 * rel
        alpha = 0.45 + 0.45 * rel
        arrow = FancyArrowPatch(
            (coords[u, 0], coords[u, 1]),
            (coords[v, 0], coords[v, 1]),
            arrowstyle="-|>",
            mutation_scale=11 + 4 * rel,
            linewidth=lw,
            color="#334155",
            alpha=alpha,
            shrinkA=7,
            shrinkB=9 if v == 0 else 7,
            zorder=2,
        )
        ax.add_patch(arrow)

    # Nodos sensores coloreados por tasa de agotamiento
    sensor_idx = np.arange(1, n)
    sensor_rates = np.array([rates[i] for i in sensor_idx], dtype=np.float64)
    ax.scatter(
        coords[sensor_idx, 0],
        coords[sensor_idx, 1],
        c=sensor_rates,
        cmap=cmap,
        norm=norm,
        s=210,
        edgecolors="#0f172a",
        linewidths=1.0,
        zorder=4,
    )

    # Resaltar nodo cuello de botella
    if bottleneck is not None:
        ax.scatter(
            [coords[bottleneck, 0]],
            [coords[bottleneck, 1]],
            s=420,
            facecolors="none",
            edgecolors="#2563eb",
            linewidths=2.4,
            linestyles="--",
            zorder=5,
            label=f"Cuello de botella (Nodo {bottleneck})",
        )

    # Gateway (nodo 0)
    ax.scatter(
        [coords[0, 0]],
        [coords[0, 1]],
        marker="*",
        s=480,
        c="#dc2626",
        edgecolors="#111827",
        linewidths=1.1,
        zorder=6,
        label="Gateway (Nodo 0)",
    )

    # Etiquetas de nodos
    for i in range(n):
        if i == 0:
            offset_y = -0.045 * instance.area_size if coords[0, 1] > 0.15 * instance.area_size else 0.045 * instance.area_size
            ax.text(
                coords[0, 0],
                coords[0, 1] + offset_y,
                "GW (0)",
                fontsize=8.8,
                fontweight="bold",
                color="#991b1b",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                zorder=7,
            )
            continue

        val_norm = norm(rates[i])
        txt_color = "white" if val_norm > 0.58 else "#0f172a"
        ax.text(
            coords[i, 0],
            coords[i, 1],
            str(i),
            fontsize=8.0,
            fontweight="bold",
            color=txt_color,
            ha="center",
            va="center",
            zorder=7,
        )
        if annotate_batteries:
            b_val = float(instance.battery_levels[i])
            b_pct = int(round(100.0 * b_val))
            # Calcular dirección alejada del vecino más cercano para evitar superposición de etiquetas
            diffs = coords[i] - coords
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nearest = int(np.argmin(dists))
            vec = coords[i] - coords[nearest]
            norm_v = float(np.linalg.norm(vec))
            if norm_v > 1e-6 and float(dists[nearest]) < 0.12 * instance.area_size:
                dir_vec = vec / norm_v
            else:
                dir_vec = np.array([0.0, -1.0])
            off = 0.048 * instance.area_size * dir_vec
            tx = float(np.clip(coords[i, 0] + off[0], -0.02 * instance.area_size, 1.02 * instance.area_size))
            ty = float(np.clip(coords[i, 1] + off[1], -0.02 * instance.area_size, 1.02 * instance.area_size))
            is_low = b_val < 0.78 or i == bottleneck
            ax.text(
                tx,
                ty,
                f"{b_pct}%",
                fontsize=6.9,
                fontweight="bold" if is_low else "normal",
                color="#991b1b" if is_low else "#334155",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    fc="#fef2f2" if is_low else "#f8fafc",
                    ec="#fca5a5" if is_low else "#cbd5e1",
                    lw=0.55,
                    alpha=0.92,
                ),
                zorder=6,
            )

    pad = 0.06 * instance.area_size
    ax.set_xlim(-pad, instance.area_size + pad)
    ax.set_ylim(-pad, instance.area_size + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("Coordenada X (m)")
    ax.set_ylabel("Coordenada Y (m)")
    ax.set_title(title, pad=10)

    metrics_text = (
        f"α máx: {audit['recomputed_objective']:.2f} μJ/ciclo\n"
        f"E_max: {audit['max_energy_uJ']:.2f} μJ | Vida: {audit['lifetime_rounds']:,.0f} ciclos\n"
        f"Fan-in máx: {audit['max_fan_in']} | Saltos máx: {audit['max_hops']} | Cuello: Nodo {bottleneck}"
    )
    ax.text(
        0.5,
        -0.18,
        metrics_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9),
    )


def plot_topology_comparison(
    instance: WSNInstance,
    base_audit: dict[str, Any],
    ext_audit: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Genera figura comparativa entre modelo Base (Avance 2) y Extendido (Etapa 3)."""
    _configure_style()
    all_rates = list(base_audit["node_depletion_rates"].values()) + list(
        ext_audit["node_depletion_rates"].values()
    )
    norm = Normalize(vmin=0.0, vmax=max(all_rates) * 1.02)
    cmap = plt.get_cmap("YlOrRd")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8))
    _draw_topology_panel(
        axes[0],
        instance,
        base_audit,
        "Modelo Base (Avance 2)\nTX constante | Sin límite fan-in | Baterías iguales",
        norm,
        cmap,
    )
    _draw_topology_panel(
        axes[1],
        instance,
        ext_audit,
        f"Modelo Extendido (Etapa 3)\nTX(d²) | K_max = {instance.K_max} | Tasa α = E_i / B_i",
        norm,
        cmap,
    )

    axes[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.20, top=0.84, wspace=0.24)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.018, 0.58])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tasa de agotamiento nodal α_i (μJ / ciclo)", fontweight="bold")

    fig.suptitle(
        f"Comparación Topológica Óptima MILP (SCIP) — Instancia N={instance.N}, semilla={instance.seed}",
        y=0.96,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_topology(
    instance: WSNInstance,
    audit: dict[str, Any],
    title: str,
    output_path: str | Path,
) -> Path:
    """Genera gráfico individual de una topología WSN auditada."""
    _configure_style()
    rates = list(audit["node_depletion_rates"].values())
    norm = Normalize(vmin=0.0, vmax=max(rates) * 1.05 if rates else 1.0)
    cmap = plt.get_cmap("YlOrRd")
    annotate_bat = bool(np.ptp(instance.battery_levels[1:]) > 1e-6)

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    _draw_topology_panel(ax, instance, audit, title, norm, cmap, annotate_batteries=annotate_bat)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.042, pad=0.04)
    cbar.set_label("Tasa de agotamiento nodal α_i (μJ / ciclo)", fontweight="bold")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scalability_time(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
    time_limit_s: float = 30.0,
) -> Path:
    """Gráfico de escalabilidad: Tiempo de ejecución (s, log) vs número de nodos N."""
    _configure_style()
    ns = [int(r["N"]) for r in summary_rows]
    milp_t = [max(1e-4, float(r["milp_tiempo_s"])) for r in summary_rows]
    greedy_t = [max(1e-4, float(r["greedy_tiempo_s"])) for r in summary_rows]
    sa_t = [max(1e-4, float(r["sa_tiempo_promedio_s"])) for r in summary_rows]
    sa_t_std = [float(r["sa_tiempo_std_s"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.plot(ns, milp_t, marker="s", linewidth=2.4, markersize=7.5, color="#b91c1c", label="MILP Exacto (SCIP)")
    ax.errorbar(
        ns,
        sa_t,
        yerr=sa_t_std,
        marker="o",
        linewidth=2.4,
        markersize=7.5,
        capsize=4,
        color="#1d4ed8",
        label="Recocido Simulado (SA, promedio ± σ)",
    )
    ax.plot(ns, greedy_t, marker="^", linewidth=2.2, markersize=7.5, color="#059669", label="Heurística Greedy")

    ax.axhline(
        time_limit_s,
        color="#64748b",
        linestyle="--",
        linewidth=1.5,
        label=f"Límite de tiempo MILP ({time_limit_s:.0f} s)",
    )

    for x_val, y_milp, row in zip(ns, milp_t, summary_rows):
        tag = "OPT" if row["milp_estado"] == "OPTIMAL" else ("LIM" if row["milp_estado"] == "FEASIBLE" else "S/S")
        y_pos = y_milp * 1.28 if y_milp < time_limit_s * 0.7 else y_milp * 1.18
        ax.text(x_val, y_pos, f"{y_milp:.2f}s ({tag})", ha="center", va="bottom", fontsize=8.5, color="#991b1b", fontweight="bold")

    for x_val, y_sa in zip(ns, sa_t):
        ax.text(x_val, y_sa * 1.28, f"{y_sa:.2f}s", ha="center", va="bottom", fontsize=8.5, color="#1e40af", fontweight="bold")

    for x_val, y_gr in zip(ns, greedy_t):
        ax.text(x_val, y_gr * 0.68, f"{y_gr*1000:.1f} ms", ha="center", va="top", fontsize=8.2, color="#065f46")

    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(2.5e-4, max(milp_t + [time_limit_s]) * 4.5)
    ax.set_xlabel("Número total de nodos de la red (N)")
    ax.set_ylabel("Tiempo de ejecución (segundos, escala logarítmica)")
    ax.set_title("Escalabilidad Computacional: MILP (SCIP) vs Greedy vs Simulated Annealing", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_objective_and_gap(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Gráfico de barras comparando objetivo alpha y brecha (Gap %) por escala N."""
    _configure_style()
    ns = [int(r["N"]) for r in summary_rows]
    x = np.arange(len(ns))
    width = 0.21

    milp_obj = [float(r["milp_objetivo"]) if r["milp_objetivo"] is not None else np.nan for r in summary_rows]
    milp_bb = [float(r["milp_cota_inferior"]) if r["milp_cota_inferior"] is not None else np.nan for r in summary_rows]
    greedy_obj = [float(r["greedy_objetivo"]) for r in summary_rows]
    sa_best = [float(r["sa_mejor_objetivo"]) for r in summary_rows]
    sa_mean = [float(r["sa_promedio_objetivo"]) for r in summary_rows]
    sa_std = [float(r["sa_std_objetivo"]) for r in summary_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.9))

    # Panel izquierdo: Objetivo alpha
    milp_plot = [0.0 if np.isnan(v) else v for v in milp_obj]
    bars_milp = ax1.bar(x - 1.5 * width, milp_plot, width, color="#dc2626", label="MILP (SCIP)")
    ax1.bar(x - 0.5 * width, greedy_obj, width, color="#10b981", label="Heurística Greedy")
    ax1.bar(x + 0.5 * width, sa_best, width, color="#1d4ed8", label="SA (Mejor)")
    ax1.bar(x + 1.5 * width, sa_mean, width, yerr=sa_std, capsize=3, color="#60a5fa", label="SA (Promedio ± σ)")

    for idx, (bar, raw_v) in enumerate(zip(bars_milp, milp_obj)):
        if np.isnan(raw_v):
            ax1.text(bar.get_x() + bar.get_width() / 2, 15, "S/S", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991b1b")

    valid_bb_x = [xi - 1.5 * width for xi, bb in zip(x, milp_bb) if not np.isnan(bb)]
    valid_bb_y = [bb for bb in milp_bb if not np.isnan(bb)]
    if valid_bb_x:
        ax1.scatter(valid_bb_x, valid_bb_y, marker="D", s=42, color="#111827", zorder=5, label="Cota dual SCIP")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Tasa máxima de agotamiento α (μJ / ciclo)")
    ax1.set_title("Calidad de Solución (menor α es mejor)")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)

    # Panel derecho: Gap (%) respecto a referencia óptima/cota dual SCIP
    gap_greedy = [float(r["greedy_gap_vs_ref_pct"]) if r["greedy_gap_vs_ref_pct"] is not None else 0.0 for r in summary_rows]
    gap_sa_best = [float(r["sa_mejor_gap_vs_ref_pct"]) if r["sa_mejor_gap_vs_ref_pct"] is not None else 0.0 for r in summary_rows]
    gap_sa_mean = [float(r["sa_promedio_gap_vs_ref_pct"]) if r["sa_promedio_gap_vs_ref_pct"] is not None else 0.0 for r in summary_rows]

    w2 = 0.27
    b_g = ax2.bar(x - w2, gap_greedy, w2, color="#10b981", label="Greedy vs Ref. SCIP")
    b_sb = ax2.bar(x, gap_sa_best, w2, color="#1d4ed8", label="SA (Mejor) vs Ref. SCIP")
    b_sm = ax2.bar(x + w2, gap_sa_mean, w2, color="#60a5fa", label="SA (Prom.) vs Ref. SCIP")

    for bars, col in [(b_g, "#065f46"), (b_sb, "#1e3a8a"), (b_sm, "#1e40af")]:
        for bar in bars:
            h = bar.get_height()
            lbl = "0%" if abs(h) < 0.05 else f"{h:.1f}%"
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(gap_greedy) * 0.015,
                lbl,
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=col,
                fontweight="bold",
            )

    x_labels = [
        f"N={r['N']}\n({'Ópt.' if r['referencia_tipo'] == 'OPTIMO_MILP' else 'Cota'})"
        for r in summary_rows
    ]
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, max(gap_greedy + [10.0]) * 1.18)
    ax2.set_ylabel("Gap respecto a referencia SCIP (%)")
    ax2.set_title("Brecha Relativa vs Óptimo / Cota Dual SCIP")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle("Comparación de Desempeño por Escala: Exacto vs Heurística vs Metaheurística", y=1.01)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sa_convergence(
    convergence_panels: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Curva de convergencia de Simulated Annealing para instancias representativas."""
    _configure_style()
    n_panels = len(convergence_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.8 * n_panels, 5.2))
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes, convergence_panels):
        hist = panel["history"]
        iters = [pt["iteration"] for pt in hist]
        cur = [pt["current_obj"] for pt in hist]
        best = [pt["best_obj"] for pt in hist]

        ax.plot(iters, cur, color="#93c5fd", linewidth=1.1, alpha=0.8, label="Estado actual SA (Metropolis)")
        ax.plot(iters, best, color="#1d4ed8", linewidth=2.5, label="Mejor acumulada SA")
        ax.axhline(
            panel["greedy_obj"],
            color="#059669",
            linestyle="--",
            linewidth=1.7,
            label=f"Greedy inicial ({panel['greedy_obj']:.2f})",
        )
        if panel.get("milp_obj") is not None:
            milp_lbl = "Óptimo MILP" if panel.get("milp_status") == "OPTIMAL" else "Incumbente MILP (30s)"
            ax.axhline(
                panel["milp_obj"],
                color="#dc2626",
                linestyle="-.",
                linewidth=1.8,
                label=f"{milp_lbl} ({panel['milp_obj']:.2f})",
            )
        if panel.get("milp_bound") is not None and panel.get("milp_status") != "OPTIMAL":
            ax.axhline(
                panel["milp_bound"],
                color="#111827",
                linestyle=":",
                linewidth=1.6,
                label=f"Cota dual SCIP ({panel['milp_bound']:.2f})",
            )

        ax.set_xlabel("Iteración")
        ax.set_ylabel("Tasa máxima de agotamiento α (μJ / ciclo)")
        ax.set_title(panel["title"])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle("Convergencia de Recocido Simulado (Simulated Annealing)", y=1.02)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_operational_scenarios(
    scenarios_data: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Gráfico de 3 paneles con las topologías de los Escenarios Operacionales A, B y C."""
    _configure_style()
    all_rates = []
    for item in scenarios_data:
        all_rates.extend(item["audit"]["node_depletion_rates"].values())
    norm = Normalize(vmin=0.0, vmax=max(all_rates) * 1.02)
    cmap = plt.get_cmap("YlOrRd")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4))
    for idx, (ax, item) in enumerate(zip(axes, scenarios_data)):
        _draw_topology_panel(
            ax,
            item["instance"],
            item["audit"],
            item["title"],
            norm,
            cmap,
            annotate_batteries=item.get("annotate_batteries", False),
        )
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)

    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.21, top=0.82, wspace=0.22)
    cbar_ax = fig.add_axes([0.92, 0.23, 0.014, 0.56])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tasa de agotamiento α_i (μJ / ciclo)", fontweight="bold")

    fig.suptitle("Evaluación de Escenarios Operacionales en Red WSN (N=25 nodos)", y=0.96)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scenarios_metrics(
    scenario_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Gráfico comparativo de tasa máxima de agotamiento y vida útil en Escenarios A, B y C."""
    _configure_style()
    labels = [r["etiqueta_corta"] for r in scenario_rows]
    x = np.arange(len(labels))
    width = 0.25

    milp_a = [float(r["milp_objetivo"]) for r in scenario_rows]
    greedy_a = [float(r["greedy_objetivo"]) for r in scenario_rows]
    sa_a = [float(r["sa_mejor_objetivo"]) for r in scenario_rows]

    milp_life = [float(r["milp_vida_util_ciclos"]) for r in scenario_rows]
    greedy_life = [float(r["greedy_vida_util_ciclos"]) for r in scenario_rows]
    sa_life = [float(r["sa_vida_util_ciclos"]) for r in scenario_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for bars_vals, offset, col, lbl in [
        (milp_a, -width, "#dc2626", "MILP (SCIP)"),
        (greedy_a, 0.0, "#10b981", "Heurística Greedy"),
        (sa_a, width, "#1d4ed8", "SA (Mejor)"),
    ]:
        bars = ax1.bar(x + offset, bars_vals, width, color=col, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max(greedy_a) * 0.015, f"{h:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(greedy_a) * 1.18)
    ax1.set_ylabel("Tasa máxima de agotamiento α (μJ / ciclo)")
    ax1.set_title("Impacto Operacional en Desgaste Pico α")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)

    for bars_vals, offset, col, lbl in [
        (milp_life, -width, "#dc2626", "MILP (SCIP)"),
        (greedy_life, 0.0, "#10b981", "Heurística Greedy"),
        (sa_life, width, "#1d4ed8", "SA (Mejor)"),
    ]:
        bars = ax2.bar(x + offset, bars_vals, width, color=col, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + max(milp_life) * 0.015, f"{h:,.0f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(milp_life + sa_life) * 1.18)
    ax2.set_ylabel("Vida útil estimada FND (ciclos, batería 1 J)")
    ax2.set_title("Vida Útil de la Red hasta Primer Nodo Agotado")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle("Comparación Cuantitativa de Escenarios Operacionales (N=25)", y=1.02)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
