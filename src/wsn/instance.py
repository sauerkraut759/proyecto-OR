"""Definición de instancias WSN y generador parametrizado con garantía de conectividad."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _resolve_traffic_vector(
    n_nodes: int,
    g: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Construye el vector de tráfico generado por nodo (g[0] = 0.0 para el Gateway)."""
    if np.isscalar(g):
        vec = np.full(n_nodes, float(g), dtype=np.float64)
        vec[0] = 0.0
        return vec

    arr = np.asarray(g, dtype=np.float64)
    if arr.shape == (n_nodes - 1,):
        vec = np.empty(n_nodes, dtype=np.float64)
        vec[0] = 0.0
        vec[1:] = arr
        return vec
    if arr.shape == (n_nodes,):
        vec = arr.copy()
        vec[0] = 0.0
        return vec
    raise ValueError(
        f"g debe ser escalar o tener largo {n_nodes - 1} o {n_nodes}; recibido shape={arr.shape}."
    )


def resolve_battery_levels(
    n_nodes: int,
    battery_levels: str | float | tuple[float, float] | Sequence[float] | np.ndarray | None,
    seed: int = 42,
) -> np.ndarray:
    """Normaliza baterías iniciales B_i por nodo (B_0 = 1.0 para el Gateway)."""
    if battery_levels is None or (isinstance(battery_levels, str) and battery_levels.lower() == "uniform"):
        return np.ones(n_nodes, dtype=np.float64)

    if np.isscalar(battery_levels):
        val = float(battery_levels)
        vec = np.full(n_nodes, val, dtype=np.float64)
        vec[0] = 1.0
        return vec

    if (
        isinstance(battery_levels, tuple)
        and len(battery_levels) == 2
        and np.isscalar(battery_levels[0])
        and np.isscalar(battery_levels[1])
        and n_nodes != 3
    ):
        low, high = float(battery_levels[0]), float(battery_levels[1])
        if low <= 0.0 or high < low:
            raise ValueError(f"Rango de baterías inválido: {(low, high)}.")
        rng = np.random.RandomState(seed + 10_000)
        vec = np.ones(n_nodes, dtype=np.float64)
        vec[1:] = rng.uniform(low, high, size=n_nodes - 1)
        return vec

    arr = np.asarray(battery_levels, dtype=np.float64)
    if arr.shape == (2,) and isinstance(battery_levels, tuple):
        low, high = float(arr[0]), float(arr[1])
        if low <= 0.0 or high < low:
            raise ValueError(f"Rango de baterías inválido: {(low, high)}.")
        rng = np.random.RandomState(seed + 10_000)
        vec = np.ones(n_nodes, dtype=np.float64)
        vec[1:] = rng.uniform(low, high, size=n_nodes - 1)
        return vec
    if arr.shape == (n_nodes - 1,):
        vec = np.ones(n_nodes, dtype=np.float64)
        vec[1:] = arr
        return vec
    if arr.shape == (n_nodes,):
        vec = arr.copy()
        vec[0] = 1.0
        return vec

    raise ValueError(
        f"battery_levels inválido (shape={arr.shape}); esperado None, (low, high) o largo {n_nodes - 1}/{n_nodes}."
    )


@dataclass
class WSNInstance:
    """Instancia parametrizada de enrutamiento en Red de Sensores Inalámbricos (WSN).

    El nodo 0 representa el Gateway (estación base sin restricción de batería ni fan-in).
    Los nodos 1..N-1 representan sensores autónomos.
    """

    N: int
    area_size: float
    D_max: float
    gateway_pos: tuple[float, float]
    seed: int
    coords: np.ndarray
    g: np.ndarray
    e_elec: float = 0.050
    eps_amp: float = 1.0e-4
    e_rx: float = 0.0828
    e_tx_base: float = 0.1104
    K_max: int | None = 5
    battery_levels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nominal_battery_uJ: float = 1_000_000.0
    name: str = ""
    generation_attempts: int = 1

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError(f"N debe ser >= 2 (al menos 1 Gateway y 1 sensor); recibido {self.N}.")
        if self.area_size <= 0.0:
            raise ValueError(f"area_size debe ser > 0; recibido {self.area_size}.")
        if self.D_max <= 0.0:
            raise ValueError(f"D_max debe ser > 0; recibido {self.D_max}.")
        if self.K_max is not None and self.K_max < 1:
            raise ValueError(f"K_max debe ser >= 1 o None; recibido {self.K_max}.")
        if self.e_elec < 0.0 or self.eps_amp < 0.0 or self.e_rx < 0.0 or self.e_tx_base < 0.0:
            raise ValueError("Los parámetros energéticos no pueden ser negativos.")
        if self.nominal_battery_uJ <= 0.0:
            raise ValueError("nominal_battery_uJ debe ser > 0.")

        gw = (float(self.gateway_pos[0]), float(self.gateway_pos[1]))
        self.gateway_pos = gw

        coords = np.asarray(self.coords, dtype=np.float64)
        if coords.shape != (self.N, 2):
            raise ValueError(f"coords debe tener shape ({self.N}, 2); recibido {coords.shape}.")
        coords = coords.copy()
        coords[0] = [gw[0], gw[1]]
        self.coords = coords

        g_vec = _resolve_traffic_vector(self.N, self.g)
        if np.any(g_vec[1:] <= 0.0):
            raise ValueError("Todo sensor i >= 1 debe generar tráfico g[i] > 0.")
        self.g = g_vec

        if self.battery_levels is None or len(self.battery_levels) == 0:
            bat = np.ones(self.N, dtype=np.float64)
        else:
            bat = resolve_battery_levels(self.N, self.battery_levels, seed=self.seed)
        if np.any(bat[1:] <= 0.0):
            raise ValueError("Todos los niveles de batería B_i deben ser > 0.")
        self.battery_levels = bat

    def dist_matrix(self) -> np.ndarray:
        """Retorna la matriz de distancias euclidianas (N x N) en metros."""
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def feasible_edges(self) -> list[tuple[int, int]]:
        """Retorna la lista de enlaces dirigidos factibles (i, j) con i >= 1, i != j, d_ij <= D_max."""
        dist = self.dist_matrix()
        edges: list[tuple[int, int]] = []
        for i in range(1, self.N):
            for j in range(self.N):
                if i != j and dist[i, j] <= self.D_max:
                    edges.append((i, j))
        return edges

    def neighbors_out(self) -> dict[int, list[int]]:
        """Retorna diccionario {sensor: [posibles_padres]} ordenados por distancia ascendente."""
        dist = self.dist_matrix()
        adj: dict[int, list[int]] = {}
        for i in range(1, self.N):
            candidates = [
                j for j in range(self.N) if i != j and dist[i, j] <= self.D_max
            ]
            candidates.sort(key=lambda j: (dist[i, j], j))
            adj[i] = candidates
        return adj

    def tx_cost_matrix(self, mode: str = "extended") -> np.ndarray:
        """Retorna matriz (N x N) de costo energético de transmisión en uJ/bit."""
        if mode == "base":
            return np.full((self.N, self.N), float(self.e_tx_base), dtype=np.float64)
        if mode == "extended":
            dist = self.dist_matrix()
            return self.e_elec + self.eps_amp * (dist ** 2)
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    def tx_energy_per_bit(self, i: int, j: int, mode: str = "extended") -> float:
        """Calcula el costo energético de transmisión de i a j en uJ/bit."""
        if mode == "base":
            return float(self.e_tx_base)
        if mode == "extended":
            d_ij = float(np.linalg.norm(self.coords[i] - self.coords[j]))
            return float(self.e_elec + self.eps_amp * (d_ij ** 2))
        raise ValueError(f"Modo inválido '{mode}'. Debe ser 'base' o 'extended'.")

    def is_connected_to_gateway(self) -> bool:
        """Verifica mediante BFS que todo sensor 1..N-1 alcanza al Gateway (nodo 0)."""
        dist = self.dist_matrix()
        visited = {0}
        queue = deque([0])
        while queue:
            curr = queue.popleft()
            for nxt in range(1, self.N):
                if nxt not in visited and dist[nxt, curr] <= self.D_max:
                    visited.add(nxt)
                    queue.append(nxt)
        return len(visited) == self.N

    def build_capacitated_tree(self, k_max: int | None = None) -> dict[int, int] | None:
        """Construye un árbol enraizado en el nodo 0 respetando D_max y fan-in <= k_max.

        Usa BFS capacitado con desempate favorable a nodos con mayor capacidad disponible
        y menor distancia al Gateway, seguido de una reparación simple de reasignación si
        algún nodo queda bloqueado por saturación de padres cercanos.
        """
        limit = self.K_max if k_max is None else k_max
        if not self.is_connected_to_gateway():
            return None

        dist = self.dist_matrix()
        adj = {
            i: [j for j in range(self.N) if i != j and dist[i, j] <= self.D_max]
            for i in range(1, self.N)
        }

        if limit is None:
            parent: dict[int, int] = {}
            queue = deque([0])
            visited = {0}
            while queue:
                curr = queue.popleft()
                for nxt in range(1, self.N):
                    if nxt not in visited and dist[nxt, curr] <= self.D_max:
                        visited.add(nxt)
                        parent[nxt] = curr
                        queue.append(nxt)
            return parent if len(parent) == self.N - 1 else None

        # Multi-ordenamiento para evitar falsos negativos cuando limit es pequeño (e.g. K_max=3)
        for strategy in ("capacity_first", "gateway_dist", "degree_first"):
            parent = {}
            children_count = [0] * self.N
            in_tree = {0}
            unassigned = set(range(1, self.N))

            while unassigned:
                frontier: list[tuple[tuple[float, ...], int, int]] = []
                for u in unassigned:
                    valid_parents = [
                        p
                        for p in adj[u]
                        if p in in_tree and (p == 0 or children_count[p] < limit)
                    ]
                    if not valid_parents:
                        continue
                    # Escoger el mejor padre actual para u
                    if strategy == "capacity_first":
                        best_p = min(
                            valid_parents,
                            key=lambda p: (
                                0 if p == 0 else 1,
                                children_count[p],
                                dist[u, p],
                                p,
                            ),
                        )
                        key = (
                            float(len(valid_parents)),
                            0.0 if best_p == 0 else float(children_count[best_p] + 1),
                            float(dist[u, best_p]),
                            float(u),
                        )
                    elif strategy == "gateway_dist":
                        best_p = min(
                            valid_parents,
                            key=lambda p: (0 if p == 0 else 1, dist[u, p], children_count[p], p),
                        )
                        key = (
                            0.0 if best_p == 0 else 1.0,
                            float(dist[u, 0]),
                            float(dist[u, best_p]),
                            float(u),
                        )
                    else:
                        best_p = min(
                            valid_parents,
                            key=lambda p: (
                                0 if p == 0 else 1,
                                children_count[p] - 0.1 * len(adj[p]),
                                dist[u, p],
                                p,
                            ),
                        )
                        key = (
                            float(len(valid_parents)),
                            -float(len(adj[u])),
                            float(dist[u, best_p]),
                            float(u),
                        )
                    frontier.append((key, u, best_p))

                if not frontier:
                    # Intentar liberar 1 cupo moviendo un nodo ya conectado hacia otro padre con cupo
                    repaired = False
                    for blocked_u in sorted(unassigned):
                        sat_parents = [p for p in adj[blocked_u] if p in in_tree and p != 0]
                        for sp in sat_parents:
                            current_kids = [k for k, par in parent.items() if par == sp]
                            for kid in current_kids:
                                alt_parents = [
                                    alt
                                    for alt in adj[kid]
                                    if alt in in_tree
                                    and alt != sp
                                    and (alt == 0 or children_count[alt] < limit)
                                ]
                                if alt_parents:
                                    alt_p = min(alt_parents, key=lambda a: (0 if a == 0 else children_count[a], dist[kid, a]))
                                    # Evitar ciclo verificando ancestros de alt_p
                                    anc = alt_p
                                    cycle = False
                                    while anc != 0:
                                        if anc == kid:
                                            cycle = True
                                            break
                                        anc = parent[anc]
                                    if not cycle:
                                        parent[kid] = alt_p
                                        children_count[sp] -= 1
                                        children_count[alt_p] += 1
                                        parent[blocked_u] = sp
                                        children_count[sp] += 1
                                        in_tree.add(blocked_u)
                                        unassigned.remove(blocked_u)
                                        repaired = True
                                        break
                            if repaired:
                                break
                        if repaired:
                            break
                    if not repaired:
                        break
                    continue

                frontier.sort(key=lambda item: item[0])
                _, chosen_u, chosen_p = frontier[0]
                parent[chosen_u] = chosen_p
                children_count[chosen_p] += 1
                in_tree.add(chosen_u)
                unassigned.remove(chosen_u)

            if not unassigned and len(parent) == self.N - 1:
                return parent

        return None

    def has_kmax_spanning_tree(self, k_max: int | None = None) -> bool:
        """Verifica si existe un árbol factible hacia el Gateway respetando fan-in <= k_max."""
        return self.build_capacitated_tree(k_max=k_max) is not None

    def with_overrides(self, **kwargs: Any) -> WSNInstance:
        """Clona la instancia preservando las coordenadas de los sensores y sobrescribiendo parámetros."""
        new_gw = kwargs.get("gateway_pos", self.gateway_pos)
        new_coords = self.coords.copy()
        new_coords[0] = [float(new_gw[0]), float(new_gw[1])]

        new_battery = kwargs.get("battery_levels", self.battery_levels)
        if new_battery is not None and not isinstance(new_battery, np.ndarray):
            new_battery = resolve_battery_levels(
                self.N,
                new_battery,
                seed=kwargs.get("seed", self.seed),
            )
        elif isinstance(new_battery, np.ndarray):
            new_battery = new_battery.copy()

        new_g = kwargs.get("g", self.g)
        if isinstance(new_g, np.ndarray):
            new_g = new_g.copy()

        return WSNInstance(
            N=kwargs.get("N", self.N),
            area_size=kwargs.get("area_size", self.area_size),
            D_max=kwargs.get("D_max", self.D_max),
            gateway_pos=(float(new_gw[0]), float(new_gw[1])),
            seed=kwargs.get("seed", self.seed),
            coords=new_coords,
            g=new_g,
            e_elec=kwargs.get("e_elec", self.e_elec),
            eps_amp=kwargs.get("eps_amp", self.eps_amp),
            e_rx=kwargs.get("e_rx", self.e_rx),
            e_tx_base=kwargs.get("e_tx_base", self.e_tx_base),
            K_max=kwargs.get("K_max", self.K_max),
            battery_levels=new_battery,
            nominal_battery_uJ=kwargs.get("nominal_battery_uJ", self.nominal_battery_uJ),
            name=kwargs.get("name", self.name),
            generation_attempts=kwargs.get("generation_attempts", self.generation_attempts),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializa la instancia a un diccionario compatible con JSON."""
        return {
            "N": int(self.N),
            "area_size": float(self.area_size),
            "D_max": float(self.D_max),
            "gateway_pos": [float(self.gateway_pos[0]), float(self.gateway_pos[1])],
            "seed": int(self.seed),
            "coords": [[float(x), float(y)] for x, y in self.coords],
            "g": [float(val) for val in self.g],
            "e_elec": float(self.e_elec),
            "eps_amp": float(self.eps_amp),
            "e_rx": float(self.e_rx),
            "e_tx_base": float(self.e_tx_base),
            "K_max": None if self.K_max is None else int(self.K_max),
            "battery_levels": [float(val) for val in self.battery_levels],
            "nominal_battery_uJ": float(self.nominal_battery_uJ),
            "name": self.name,
            "generation_attempts": int(self.generation_attempts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WSNInstance:
        """Reconstruye una WSNInstance desde un diccionario JSON."""
        return cls(
            N=int(data["N"]),
            area_size=float(data["area_size"]),
            D_max=float(data["D_max"]),
            gateway_pos=(float(data["gateway_pos"][0]), float(data["gateway_pos"][1])),
            seed=int(data["seed"]),
            coords=np.asarray(data["coords"], dtype=np.float64),
            g=np.asarray(data["g"], dtype=np.float64),
            e_elec=float(data.get("e_elec", 0.050)),
            eps_amp=float(data.get("eps_amp", 1.0e-4)),
            e_rx=float(data.get("e_rx", 0.0828)),
            e_tx_base=float(data.get("e_tx_base", 0.1104)),
            K_max=None if data.get("K_max") is None else int(data["K_max"]),
            battery_levels=np.asarray(data.get("battery_levels", [1.0] * int(data["N"])), dtype=np.float64),
            nominal_battery_uJ=float(data.get("nominal_battery_uJ", 1_000_000.0)),
            name=str(data.get("name", "")),
            generation_attempts=int(data.get("generation_attempts", 1)),
        )

    def save_json(self, path: str | Path) -> Path:
        """Guarda la instancia en un archivo JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return out_path

    @classmethod
    def load_json(cls, path: str | Path) -> WSNInstance:
        """Carga una instancia desde un archivo JSON."""
        in_path = Path(path)
        with in_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)


def generate_instance(
    N: int = 15,
    area_size: float = 70.0,
    D_max: float = 40.0,
    gateway_pos: tuple[float, float] | Sequence[float] | None = None,
    seed: int = 42,
    g: float | Sequence[float] | np.ndarray = 400.0,
    e_elec: float = 0.050,
    eps_amp: float = 1.0e-4,
    e_rx: float = 0.0828,
    K_max: int | None = 5,
    battery_levels: str | float | tuple[float, float] | Sequence[float] | np.ndarray | None = None,
    *,
    e_tx_base: float = 0.1104,
    nominal_battery_uJ: float = 1_000_000.0,
    name: str = "",
    max_attempts: int = 500,
    ensure_kmax_feasible: bool = True,
) -> WSNInstance:
    """Genera una instancia sintética WSN garantizando conectividad hacia el Gateway (nodo 0).

    En el intento 0 usa `np.random.RandomState(seed)` para reproducir exactamente la
    disposición de Avance 2 (`N=15, area_size=70, D_max=40, seed=42`). Si el grafo
    resultante no es conexo (o no admite árbol con fan-in <= K_max), regenera con
    semillas deterministas derivadas hasta encontrar una topología factible.
    """
    if gateway_pos is None:
        gw = (area_size / 2.0, area_size / 2.0)
    else:
        gw = (float(gateway_pos[0]), float(gateway_pos[1]))

    g_vec = _resolve_traffic_vector(N, g)
    bat_vec = resolve_battery_levels(N, battery_levels, seed=seed)

    for attempt in range(max_attempts):
        attempt_seed = seed if attempt == 0 else seed + attempt * 1009
        rng = np.random.RandomState(attempt_seed)
        coords = rng.rand(N, 2) * float(area_size)
        coords[0] = [gw[0], gw[1]]

        candidate = WSNInstance(
            N=N,
            area_size=float(area_size),
            D_max=float(D_max),
            gateway_pos=gw,
            seed=int(seed),
            coords=coords,
            g=g_vec,
            e_elec=float(e_elec),
            eps_amp=float(eps_amp),
            e_rx=float(e_rx),
            e_tx_base=float(e_tx_base),
            K_max=K_max,
            battery_levels=bat_vec,
            nominal_battery_uJ=float(nominal_battery_uJ),
            name=name or f"wsn_N{N}_seed{seed}",
            generation_attempts=attempt + 1,
        )

        if not candidate.is_connected_to_gateway():
            continue
        if ensure_kmax_feasible and K_max is not None and not candidate.has_kmax_spanning_tree(K_max):
            continue
        return candidate

    raise ValueError(
        f"No se pudo generar una instancia conexa para N={N}, area_size={area_size}, "
        f"D_max={D_max}, K_max={K_max} tras {max_attempts} intentos."
    )
