import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp

def resolver_ruteo_wsn():
    # ==========================================
    # 1. PARÁMETROS DEL SISTEMA (Instancia)
    # ==========================================
    N = 15  # Número de sensores + 1 Gateway
    D_max = 40.0  # Rango de cobertura en metros
    e_tx = 0.1104  # micro-Joules / bit
    e_rx = 0.0828  # micro-Joules / bit
    g = 400  # Tráfico generado por nodo en bits
    
    # Generar coordenadas aleatorias en un área de 70x70 metros
    np.random.seed(42)  # Semilla para reproducibilidad
    coords = np.random.rand(N, 2) * 70 
    # Forzamos al Gateway (nodo 0) a estar en el centro
    coords[0] = [35.0, 35.0]

    # Calcular matriz de distancias
    dist = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    
    # Conjunto de enlaces factibles E: (i, j) donde i != j y distancia <= D_max
    # i debe ser > 0 (el Gateway no transmite, solo recibe)
    E = [(i, j) for i in range(1, N) for j in range(N) if i != j and dist[i, j] <= D_max]
    
    # Big-M: Tráfico máximo posible en un enlace (todos los sensores pasando por un solo cable)
    M = (N - 1) * g

    # ==========================================
    # 2. INICIALIZAR EL SOLVER (SCIP)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver no está disponible.")
        return

    # ==========================================
    # 3. VARIABLES DE DECISIÓN
    # ==========================================
    x = {} # Variable binaria de topología
    f = {} # Variable continua de flujo
    
    for (i, j) in E:
        x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
        f[i, j] = solver.NumVar(0, solver.infinity(), f'f_{i}_{j}')
        
    E_max = solver.NumVar(0, solver.infinity(), 'E_max')

    # ==========================================
    # 4. RESTRICCIONES
    # ==========================================
    # Familia 1: Asignación (Cada sensor se conecta a exactamente 1 destino)
    for i in range(1, N):
        solver.Add(solver.Sum([x[i, j] for j in range(N) if (i, j) in E]) == 1)

    # Familia 2: Balance de flujo (Lo que entra + lo que genero = lo que sale)
    for i in range(1, N):
        flujo_entrante = solver.Sum([f[k, i] for k in range(1, N) if (k, i) in E])
        flujo_saliente = solver.Sum([f[i, j] for j in range(N) if (i, j) in E])
        solver.Add(flujo_entrante + g == flujo_saliente)

    # Familia 3: Big-M (Flujo solo si el enlace existe)
    for (i, j) in E:
        solver.Add(f[i, j] <= M * x[i, j])

    # Familia 4: Min-Max Energía (E_max debe ser >= al consumo de cada nodo individual)
    for i in range(1, N):
        consumo_rx = e_rx * solver.Sum([f[k, i] for k in range(1, N) if (k, i) in E])
        consumo_tx = e_tx * solver.Sum([f[i, j] for j in range(N) if (i, j) in E])
        solver.Add(consumo_rx + consumo_tx <= E_max)

    # ==========================================
    # 5. FUNCIÓN OBJETIVO
    # ==========================================
    solver.Minimize(E_max)

    # ==========================================
    # 6. RESOLUCIÓN Y EXTRACCIÓN DE RESULTADOS
    # ==========================================
    print("Iniciando resolución matemática...")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("\n¡Solución Óptima Encontrada!")
        print(f"Consumo de energía del nodo más estresado (E_max): {E_max.solution_value():.2f} micro-Joules")
        
        # Extraer enlaces activos
        enlaces_activos = [(i, j) for (i, j) in E if x[i, j].solution_value() > 0.5]
        
        # Graficar
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        G.add_edges_from(enlaces_activos)
        
        posiciones = {i: coords[i] for i in range(N)}
        colores = ['red' if i == 0 else 'lightblue' for i in range(N)]
        tamanos = [600 if i == 0 else 300 for i in range(N)]
        
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos=posiciones, with_labels=True, node_color=colores, 
                node_size=tamanos, font_weight='bold', arrows=True, 
                arrowsize=15, edge_color='gray')
        plt.title(f"Topología Óptima WSN - Consumo Min-Max: {E_max.solution_value():.2f} uJ")
        plt.plot(35, 35, marker='x', markersize=10, color='red', label="Gateway (Nodo 0)")
        plt.legend()
        plt.show()
    else:
        print("El solver no pudo encontrar una solución óptima. Revisa si D_max es muy pequeño y hay nodos aislados.")

if __name__ == '__main__':
    resolver_ruteo_wsn()
