# Proyecto Optimización


La decisión central que aborda este proyecto consiste en determinar la topología lógica de la red y el enrutamiento del tráfico, es decir, definir a qué nodo vecino debe transmitir sus datos cada sensor en cada etapa de la comunicación.

Esta decisión es crítica debido al fenómeno conocido como "agotamiento por retransmisión" en las proximidades de la estación central (hotspot / bottleneck problem). Los sensores ubicados cerca del Gateway no solo transmiten sus mediciones propias, sino que deben retransmitir el flujo acumulado de gran parte de la red. Como resultado, estos nodos consumen su batería a una tasa acelerada, provocando la desconexión prematura de regiones enteras de la red, incluso si los nodos periféricos aún conservan niveles elevados de energía.

Optimizar la estructura de enlaces para balancear la carga energética entre todos los nodos es, por tanto, una decisión compleja e indispensable para garantizar la operatividad continua del sistema.

El proyecto contempla el diseño y formulación matemática formal del problema de enrutamiento y asignación de flujo mediante un modelo de Programación Lineal Entera Mixta (MILP). Incluye su implementación en un entorno computacional (Python con solvers de optimización), la generación de instancias sintéticas de prueba y la evaluación cuantitativa de escenarios comparativos (variación en posición de Gateway y densidad de nodos).
