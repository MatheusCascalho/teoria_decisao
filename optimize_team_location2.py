import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from dataclasses import dataclass, field

# Estrutura de dados para definir parâmetros do problema
@dataclass
class problemDefinition:
    distance_matrix: pd.DataFrame
    base_map: dict
    ativo_map: dict
    ETA: float = 0.2         # Percentual para balanceamento de ativos (𝜂/m)
    n_equipes: int = 3       # Número de equipes
    n_ativos: int = 125      # Número de ativos
    n_bases: int = 14        # Número de bases

# Estrutura de dados para representar uma solução do problema
@dataclass
class solution:
    ativo_equipe: list       # Atribuição de ativos para equipes
    equipe_base: list        # Atribuição de bases para equipes
    fitness: float = 0       # Fitness da solução
    penalidade: float = 0    # Penalidade da solução

    @property
    def fitness_penalizado(self):
        return self.penalidade + self.fitness

    @property
    def ativo_base(self):
        # Mapeia cada ativo para a base correspondente da equipe
        return [self.equipe_base[equipe] for equipe in self.ativo_equipe]

# Estrutura para armazenar o histórico das soluções
@dataclass
class history:
    fit: list = field(default_factory=list)
    sol: list = field(default_factory=list)
    pen: list = field(default_factory=list)
    fit_pen: list = field(default_factory=list)

    def update(self, solution: solution):
        self.fit.append(solution.fitness)
        self.sol.append(solution)
        self.pen.append(solution.penalidade)
        self.fit_pen.append(solution.fitness_penalizado)

    @property
    def best_solution(self):
        return self.sol[-1]  # Última solução é a melhor

    @property
    def is_stable(self):
        n_sol = len(self.sol)
        min_stable = int(n_sol * 0.2)
        return n_sol >= 200 and all(s.fitness_penalizado == self.sol[-1].fitness_penalizado for s in self.sol[-min_stable:])

# Função para gerar uma solução inicial
def sol_inicial(prob_def: problemDefinition, apply_constructive_heuristic=True, use_random=True):
    if not apply_constructive_heuristic:
        # Solução inicial aleatória
        y = np.random.randint(0, prob_def.n_bases, size=prob_def.n_equipes)
        h = np.random.randint(0, prob_def.n_equipes, size=prob_def.n_ativos)
        return solution(equipe_base=y, ativo_equipe=h)

    # Solução inicial com heurística construtiva
    inverse_ativo_map = {v: k for k, v in prob_def.ativo_map.items()}
    df = pd.DataFrame(inverse_ativo_map).T.rename(columns={0: 'latitude', 1: 'longitude'})
    df['type'] = 'ativo'

    inverse_base_map = {v: k for k, v in prob_def.base_map.items()}
    df_base = pd.DataFrame(inverse_base_map).T.rename(columns={0: 'latitude', 1: 'longitude'})
    df_base['type'] = 'base'
    df = pd.concat([df, df_base])

    # Clustering para definir bases das equipes
    kmeans = KMeans(n_clusters=prob_def.n_equipes, random_state=0).fit(df[['latitude', 'longitude']])
    df['cluster'] = kmeans.labels_

    equipe_base = []
    ativos_equipe = np.zeros(prob_def.n_ativos)
    bases = df[df['type'] == 'base'].reset_index()

    for i, centroid in enumerate(kmeans.cluster_centers_):
        cluster_points = bases[bases['cluster'] == i]
        closest_index = cluster_points.index[np.random.randint(len(cluster_points))] if use_random else cluster_points.apply(
            lambda row: np.linalg.norm([row['latitude'] - centroid[0], row['longitude'] - centroid[1]]), axis=1).idxmin()
        
        equipe_base.append(closest_index)
        ativos_equipe[df[(df['type'] == 'ativo') & (df['cluster'] == i)].index] = i

    return solution(ativo_equipe=list(ativos_equipe.astype(int)), equipe_base=equipe_base)

# Função para calcular penalidade de uma solução
def get_penalidade(solution: solution, prob_def: problemDefinition):
    ativos_por_equipe = [np.sum(np.array(solution.ativo_equipe) == k) for k in range(prob_def.n_equipes)]
    penalidade = 0
    min_ativos = prob_def.ETA * prob_def.n_ativos / prob_def.n_equipes
    for qtd_ativos in ativos_por_equipe:
        penalidade += 100 * max(0, min_ativos - qtd_ativos) ** 2
    return penalidade

# Função para balancear ativos entre equipes
def equilibrio_ativos(solution: solution, prob_def: problemDefinition):
    ativos_por_equipe = [np.sum(np.array(solution.ativo_equipe) == k) for k in range(prob_def.n_equipes)]
    solution.fitness = max(ativos_por_equipe) - min(ativos_por_equipe)
    solution.penalidade = get_penalidade(solution, prob_def)
    return solution

# Função objetivo para minimizar distâncias entre ativos e bases
def minimiza_distancias(solution: solution, prob_def: problemDefinition):
    total_distance = sum(prob_def.distance_matrix.loc[ativo, solution.ativo_base[ativo]] for ativo in range(prob_def.n_ativos))
    solution.fitness = total_distance
    solution.penalidade = get_penalidade(solution, prob_def)
    return solution

# Função de perturbação (shake) para explorar a vizinhança de uma solução
def shake(solution: solution, k: int, prob_def: problemDefinition):
    neighbor = copy.deepcopy(solution)
    r_equipe, r_ativo = np.random.randint(prob_def.n_equipes), np.random.randint(prob_def.n_ativos)

    if k == 1:
        neighbor.equipe_base[r_equipe] = (neighbor.equipe_base[r_equipe] + 1) % prob_def.n_bases
    elif k == 2:
        neighbor.ativo_equipe[r_ativo] = (neighbor.ativo_equipe[r_ativo] + 1) % prob_def.n_equipes
    elif k == 3:
        neighbor.equipe_base[r_equipe] = (neighbor.equipe_base[r_equipe] + 1) % prob_def.n_bases
        neighbor.ativo_equipe[r_ativo] = (neighbor.ativo_equipe[r_ativo] + 1) % prob_def.n_equipes

    return neighbor

# Função para primeira melhoria na vizinhança
def first_improvement(solution: solution, k: int, objective_function: callable, prob_def, max_iteration=2e6):
    current_fitness = objective_function(solution, prob_def).fitness_penalizado
    max_neighbors = {1: 3, 2: 125, 3: 20}
    
    for _ in range(int(max_neighbors.get(k, max_iteration))):
        neighbor = shake(solution, k, prob_def)
        if objective_function(neighbor, prob_def).fitness_penalizado < current_fitness:
            return neighbor
    return solution

# Função para troca de vizinhança
def neighborhoodChange(current_solution, candidate_solution, k):
    return (copy.deepcopy(candidate_solution), 1) if candidate_solution.fitness_penalizado < current_solution.fitness_penalizado else (current_solution, k + 1)

# Implementação da VNS
def RVNS(prob_def, initial_solution, objective_function, max_iteration, historico, kmax=3):
    current_solution = initial_solution
    for _ in range(int(max_iteration)):
        k = 1
        while k <= kmax:
            candidate_solution = objective_function(shake(current_solution, k, prob_def), prob_def)
            current_solution, k = neighborhoodChange(current_solution, candidate_solution, k)
            historico.update(current_solution)
    return historico

def BasicVNS(prob_def, initial_solution, objective_function, max_iteration, historico, kmax=3):
    current_solution = initial_solution
    for _ in range(int(max_iteration)):
        k = 1
        while k < kmax:
            candidate_solution = first_improvement(current_solution, k, objective_function, prob_def)
            candidate_solution = objective_function(candidate_solution, prob_def)
            current_solution, k = neighborhoodChange(current_solution, candidate_solution, k)
            historico.update(current_solution)
        if historico.is_stable:
            break
    return historico



def get_problem_definition():
    data = pd.read_csv("probdata.csv", delimiter=";", header=None, decimal=',', names=[
        "Latitude_Base", "Longitude_Base", "Latitude_Ativo", "Longitude_Ativo", "Distância"
    ])
    data.head()
    ativos = data[['Latitude_Ativo', 'Longitude_Ativo']].drop_duplicates().reset_index()
    ativos_map = {(r['Latitude_Ativo'], r['Longitude_Ativo']): i for i, r in ativos.iterrows()}

    bases = data[['Latitude_Base', 'Longitude_Base']].drop_duplicates().reset_index()
    bases_map = {(r['Latitude_Base'], r['Longitude_Base']): i for i, r in bases.iterrows()}
    bases_map

    data['ativo'] = data[['Latitude_Ativo', 'Longitude_Ativo']].apply(
        lambda r: ativos_map.get((r['Latitude_Ativo'], r['Longitude_Ativo'])), axis=1)
    data['base'] = data[['Latitude_Base', 'Longitude_Base']].apply(
        lambda r: bases_map.get((r['Latitude_Base'], r['Longitude_Base'])), axis=1)
    distance_matrix = data.set_index(['ativo', 'base'])[['Distância']].unstack(1).fillna(0)
    distance_matrix.columns = distance_matrix.columns.droplevel(0)

    prob_def = problemDefinition(
        base_map=bases_map,
        ativo_map=ativos_map,
        n_bases=len(bases_map),
        n_ativos=len(ativos_map),  # Número de ativos,
        n_equipes=3,
        distance_matrix=distance_matrix
    )
    return prob_def
