# Optimización de Topología y Enrutamiento Energéticamente Eficiente en WSN (Etapa 3 — ELI148)

Proyecto semestral de **Técnicas de Optimización (ELI148)** sobre diseño de topologías tipo árbol y asignación de flujo multi-salto en **Redes de Sensores Inalámbricos (WSN)** bajo el estándar IEEE 802.15.4 / Zigbee (referencia de hardware *Texas Instruments CC2652R*).

La decisión central consiste en determinar a qué nodo padre transmite cada sensor para mitigar el **agotamiento prematuro por retransmisión** (*hotspot / bottleneck problem*) en las cercanías del Gateway (nodo `0`), maximizando la vida útil de la red hasta la desconexión del primer nodo (*First Node Dies*, FND).

---

## Formulación Matemática: Modelo Base vs. Extendido

### Conjuntos y Parámetros
- **Nodos**: $V = \{0, 1, \dots, N-1\}$, donde `0` es el Gateway y $S = V \setminus \{0\}$ son los sensores.
- **Enlaces factibles**: $E = \{(i, j) \in S \times V \mid i \neq j \land d_{ij} \le D_{\max}\}$.
- **Tráfico generado**: $g_i = 400\ \text{bits/ciclo}$ por sensor ($g_0 = 0$).
- **Costo energético de recepción**: $e_{\text{rx}} = 0.0828\ \mu\text{J/bit}$.
- **Costo energético de transmisión**:
  - **Modo Base (Avance 2)**: constante $e_{\text{tx}} = 0.1104\ \mu\text{J/bit}$, baterías homogéneas $B_i = 1.0$, sin límite de fan-in.
  - **Modo Extendido (Etapa 3)**: dependiente de la distancia euclidiana $e_{\text{tx}}(d_{ij}) = e_{\text{elec}} + \varepsilon_{\text{amp}}\, d_{ij}^2$ (con $e_{\text{elec}} = 0.050\ \mu\text{J/bit}$ y $\varepsilon_{\text{amp}} = 10^{-4}\ \mu\text{J/(bit}\cdot\text{m}^2)$), límite de fan-in por sensor repetidor $K_{\max}$ y baterías residuales normalizadas $B_i \in (0, 1]$.

### Variables de Decisión y Objetivo
- $x_{ij} \in \{0, 1\}$: activación del enlace dirigido $(i, j) \in E$.
- $f_{ij} \ge 0$: flujo de datos (bits/ciclo) que transita por $(i, j) \in E$.
- $\alpha \ge 0$: tasa máxima de agotamiento energético relativo de la red:

$$\min Z = \alpha$$

### Restricciones
1. **Asignación única (árbol hacia el Gateway)**: $\sum_{j:(i,j)\in E} x_{ij} = 1 \quad \forall i \in S$
2. **Conservación de flujo**: $\sum_{k:(k,i)\in E} f_{ki} + g_i = \sum_{j:(i,j)\in E} f_{ij} \quad \forall i \in S$
3. **Acoplamiento lógico ajustado**: $g_i\, x_{ij} \le f_{ij} \le M_{ij}\, x_{ij} \quad \forall (i, j) \in E$, donde $M_{ij} = \sum_{s \in S} g_s$ si $j = 0$ y $\sum_{s \in S} g_s - g_j$ si $j \in S$.
4. **Cota Min-Max de desgaste relativo**:
   $$e_{\text{rx}} \sum_{k:(k,i)\in E} f_{ki} + \sum_{j:(i,j)\in E} e_{\text{tx}}(d_{ij})\, f_{ij} \le B_i\, \alpha \quad \forall i \in S$$
5. **Capacidad máxima de retransmisión / fan-in (solo modo extendido)**:
   $$\sum_{k:(k,i)\in E} x_{ki} \le K_{\max} \quad \forall i \in S$$

La vida útil estimada FND (en ciclos) respecto a una batería nominal de $1\ \text{J} = 10^6\ \mu\text{J}$ corresponde a $10^6 / \alpha$.

---

## Arquitectura del Repositorio

```text
proyecto-OR/
├── main.py                              # Punto de entrada CLI limpio
├── pyproject.toml                       # Dependencias y configuración uv_build
├── README.md                            # Documentación y guía reproducible
├── src/
│   └── wsn/
│       ├── __init__.py                  # Exportaciones públicas del paquete wsn
│       ├── instance.py                  # Dataclass WSNInstance, generador conexo y JSON I/O
│       ├── exact_solver.py              # Modelo MILP exacto en OR-Tools (SCIP)
│       ├── validator.py                 # Auditor independiente de factibilidad y objetivo
│       ├── metaheuristic.py             # Heurística Greedy balanceada y Simulated Annealing
│       ├── experiments.py               # Suite reproducible (Exp 1, Exp 2 y Exp 3)
│       ├── plotting.py                  # Gráficos en español a 300 DPI para diapositivas
│       └── cli.py                       # Subcomandos y argumentos CLI
└── results/
    ├── instances/                       # Instancias sintéticas exportadas en JSON
    ├── tables/                          # Tablas de resultados en CSV
    └── figures/                         # Figuras PNG de alta resolución (300 DPI)
```

---

## Instalación y Ejecución Reproducible

Requiere Python $\ge 3.11$ y [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Instalar dependencias en el entorno virtual gestionado por uv
uv sync

# 2. Ejecutar la suite experimental completa (Exp 1, Exp 2 y Exp 3)
uv run python main.py --all
```

*(En entornos sin conexión tras sincronizar dependencias: `uv run --offline python main.py --all`.)*

### Otros Comandos Útiles
```bash
# Ejecutar experimentos individuales
uv run python main.py --exp1
uv run python main.py --exp2 --sa-runs 5 --time-limit 30
uv run python main.py --exp3

# Resolver una instancia personalizada o cargada desde JSON
uv run python main.py --solve-single --nodes 25 --area 90 --dmax 40 --kmax 3 --battery-range 0.6 1.0 --solver all
uv run python main.py --solve-single --instance results/instances/exp1_n15_seed42.json --mode extended --solver milp
```

---

## Artefactos Generados y Validación

- **Tablas CSV (`results/tables/`)**:
  - `exp1_base_vs_extended.csv`: verificación en $N=15, \text{seed}=42$ (Base Avance 2 = $121.44\ \mu\text{J/ciclo}$ vs. Extendido = $94.10\ \mu\text{J/ciclo}$).
  - `exp2_scalability_summary.csv` y `exp2_sa_runs_detail.csv`: escalabilidad en $N \in [15, 25, 40, 60, 100]$ comparando MILP (`SCIP`, límite 30 s), Greedy y 5 corridas independientes de Simulated Annealing.
  - `exp3_operational_scenarios.csv`: comparación en $N=25$ sobre los 3 escenarios operacionales: **(A)** Gateway centrado `(45, 45)`, **(B)** Gateway en esquina `(5, 5)` y **(C)** restricción estricta $K_{\max}=3$ con baterías heterogéneas $B_i \in [0.6, 1.0]$.
- **Figuras PNG (`results/figures/`, 300 DPI)**:
  - `exp1_topologia_base_vs_extendido.png`
  - `exp2_escalabilidad_tiempo.png`
  - `exp2_objetivo_y_gap.png`
  - `exp2_convergencia_sa.png`
  - `exp3_escenarios_operacionales.png`
  - `exp3_comparacion_escenarios.png`
- **Auditoría independiente (`wsn.validator.validate_solution`)**: toda solución exacta y metaheurística se audita de forma independiente verificando ausencia de ciclos, conectividad total al Gateway (nodo `0`), respeto de $D_{\max}$ y $K_{\max}$, grado de salida igual a `1` y recálculo exacto de flujos y consumos energéticos (`is_feasible=True` en el 100% de las corridas).
