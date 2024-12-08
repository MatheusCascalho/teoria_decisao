from copy import deepcopy

import pandas as pd
import seaborn as sns

'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from tqdm import tqdm

'''
Define um tipo de dado similar ao Pascal "record" or C "struct"
'''
class Struct:
    pass


from dataclasses import dataclass, field


@dataclass
class problem_definition:
    # Definir parâmetros
    distance_matrix: pd.DataFrame
    base_map: dict
    ativo_map: dict
    ETA: float = 0.2  # Percentual para balanceamento de ativos (𝜂/m)
    n_equipes: int = 3  # Quantidade de equipes
    n_ativos: int = 125
    n_bases: int = 14

@dataclass
class solution:
    ativo_equipe: list # h
    equipe_base: list # y
    fitness: float = 0
    penalidade: float = 0
    multi_fitness: dict = field(default_factory=dict)

    @property
    def fitness_penalizado(self):
        return self.penalidade + self.fitness

    @property
    def ativo_base(self): # x
        x = []
        for equipe in self.ativo_equipe:
            base = self.equipe_base[equipe]
            x.append(base)
        return x


@dataclass
class history:
    min_iterations: int
    fit: list = field(default_factory=list)
    sol: list = field(default_factory=list)
    pen: list = field(default_factory=list)
    fit_pen: list = field(default_factory=list)

    def update(self, x: solution):
        self.fit.append(x.fitness)
        self.sol.append(x)
        self.pen.append(x.penalidade)
        self.fit_pen.append(x.fitness_penalizado)

    @property
    def best_solution(self):
        return self.sol[-1]

    @property
    def is_stable(self):
        n_sol = len(self.sol)
        min_stabel = int(n_sol*0.2)
        return n_sol>=self.min_iterations and all(s.fitness_penalizado == self.sol[-1].fitness_penalizado for s in self.sol[-min_stabel:])

    @property
    def is_locked(self):
        return len(self.sol) > 20 and  all(s.fitness_penalizado == self.sol[-1].fitness_penalizado for s in self.sol[-20:])

'''
Implementa uma solução inicial para o problema
'''
def sol_inicial(prob_def: problem_definition, apply_constructive_heuristic=True, use_random=True):
    '''
    Modelou-se uma solução x como um vetor binário

    x = [x1 x2 ... xn]
    '''

    if apply_constructive_heuristic == False:
        # Constrói solução inicial aleatoriamente
        y = np.random.randint(0, prob_def.n_bases, size=prob_def.n_equipes)
        h = np.random.randint(0, prob_def.n_equipes, size=prob_def.n_ativos)
        sol = solution(equipe_base=y, ativo_equipe=h)
    else:
        # Definindo base das equipes
        inverse_ativo_map = {v: k for k, v in prob_def.ativo_map.items()}
        df = pd.DataFrame(inverse_ativo_map).T
        df.columns = ['latitude', 'longitude']
        df['type'] = 'ativo'

        inverse_base_map = {v: k for k, v in prob_def.base_map.items()}
        df_base = pd.DataFrame(inverse_base_map).T
        df_base.columns = ['latitude', 'longitude']
        df_base['type'] = 'base'
        df = pd.concat([df, df_base])

        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(df[['latitude', 'longitude']])

        # Adicionando a coluna de labels ao DataFrame
        df['cluster'] = kmeans.labels_

        # Obtendo os centróides
        centroids = kmeans.cluster_centers_
        # Encontrando o ponto mais próximo de cada centróide
        equipe_base = []
        ativos_equipe = np.zeros(prob_def.n_ativos)
        bases = df[df['type']=='base'].reset_index()
        for i, centroid in enumerate(centroids):
            # Filtrar apenas os pontos do cluster atual
            cluster_points = bases[bases['cluster'] == i]
            if not use_random:
                # Calcular a distância de cada ponto ao centróide atual
                distances = np.sqrt(
                    (cluster_points['latitude'] - centroid[0]) ** 2 + (cluster_points['longitude'] - centroid[1]) ** 2)

                # Obter o índice do ponto mais próximo
                closest_index = distances.idxmin()
            else:
                closest_index = list(cluster_points.index)[np.random.randint(len(cluster_points))]

            # Adicionar o ponto mais próximo à lista
            equipe_base.append(closest_index)
            ativos = df[(df['type'] == 'ativo')&(df['cluster'] == i)].index
            ativos_equipe[ativos] = i

        sol = solution(
            ativo_equipe=list(ativos_equipe.astype(int)),
            equipe_base=equipe_base
        )

    return sol

def get_penalidade(x: solution, prob_def: problem_definition):
    ativos_por_equipe = [len(np.where(np.array(x.ativo_equipe) == k)[0]) for k in range(prob_def.n_equipes)]
    penalidade = 0
    min_ativos = prob_def.ETA * prob_def.n_ativos / prob_def.n_equipes
    # print(f"Minimo de ativos:{min_ativos}")
    u = 100
    for equipe, qtd_ativos in enumerate(ativos_por_equipe):
        g = min_ativos - qtd_ativos
        # print(f"Qtd. Ativos: {qtd_ativos} \t G: {g}")
        penalidade += u * max(0, g) ** 2
    return penalidade

def equilibrio_ativos(x: solution, prob_def: problem_definition):
    ativos_por_equipe = [len(np.where(np.array(x.ativo_equipe) == k)[0]) for k in range(prob_def.n_equipes)]

    x.fitness = max(ativos_por_equipe) - min(ativos_por_equipe)
    x.penalidade = get_penalidade(x, prob_def)

    return x

def minimiza_distancia_maxima(x: solution, prob_def: problem_definition):
    distancias = []
    for ativo, base in enumerate(x.ativo_base):
        d = prob_def.distance_matrix.loc[ativo, base]
        distancias.append(d)

    x.fitness = max(distancias)
    x.penalidade = get_penalidade(x, prob_def)

    return x


'''
Implementa a função objetivo do problema
'''
def minimiza_distancias(x: solution, prob_def: problem_definition):
    '''
    x = [x1 x2 ... xn]
    '''

    ativo_base = x.ativo_base
    total_distance = 0
    for ativo, base in enumerate(ativo_base):
        d = prob_def.distance_matrix.loc[ativo, base]
        total_distance += d

    x.fitness = total_distance

    x.penalidade = get_penalidade(x, prob_def)

    return x


class WeightedSum:
    def __init__(self, step=0.05):
        self.step = step
        self.weights = [1,0]
        self.histories = []

    def weighted_sum(self, x: solution, prob_def: problem_definition):
        new_x = deepcopy(x)
        f1 = minimiza_distancias(new_x, prob_def).fitness_penalizado
        f2 = minimiza_distancia_maxima(new_x, prob_def).fitness_penalizado
        min_f2 = prob_def.distance_matrix.min().min()
        max_f2 = prob_def.distance_matrix.max().max()
        min_f1 = sum(sorted(prob_def.distance_matrix.values.reshape((125*14)))[:prob_def.n_ativos])
        max_f1 = sum(np.abs(sorted(prob_def.distance_matrix.values.reshape((125*14))*-1))[:prob_def.n_ativos])

        f2_norm = (f2 - min_f2) / (max_f2 - min_f2)
        f1_norm = (f1-min_f1)/(max_f1-min_f1)
        fit = self.weights[0]*f1_norm + self.weights[1]*f2_norm

        new_x.fitness = fit
        new_x.multi_fitness = {"f1": f1, "f2": f2, "pond": fit, "f1_norm": f1_norm, "f2_norm": f2_norm}
        new_x.penalidade = get_penalidade(new_x, prob_def)
        return new_x

    def balance(self):
        self.weights[0] -= self.step
        self.weights[1] += self.step
        if self.weights[1]>1:
            raise Exception('All weights tried')


'''
Implementa a função shake
'''
def shake(x: solution, k: int, prob_def: problem_definition):
    y = copy.deepcopy(x)
    r_equipe = np.random.randint(prob_def.n_equipes)
    r_ativos = np.random.permutation(prob_def.n_ativos)

    # trocando ativo de equipe - 125 vizinhos
    if k == 1:
        # worst_distances = np.where(prob_def.distance_matrix.loc[:,y.equipe_base] == prob_def.distance_matrix.loc[:,y.equipe_base].max())
        # ativos, bases = worst_distances
        r_ativo = r_ativos[0]
        y.ativo_equipe[r_ativo] = np.random.randint(prob_def.n_equipes)
        # if y.ativo_equipe[r_ativo] == prob_def.n_equipes - 1:
        #     y.ativo_equipe[r_ativo] = 0
        # else:
        #     y.ativo_equipe[r_ativo] += 1

    # trocando equipe de base - 3 vizinhos
    elif k == 2:
        # y.solution[r[0]] = not(y.solution[r[0]])
        if y.equipe_base[r_equipe] == prob_def.n_bases - 1:
            y.equipe_base[r_equipe] = 0
        else:
            y.equipe_base[r_equipe] += 1



    # trocando ativo de equipe e equipe de base - 1.953.125 vizinhos
    elif k == 3:
        # if y.equipe_base[r_equipe] == prob_def.n_bases - 1:
        y.equipe_base[r_equipe] = np.random.randint(prob_def.n_bases)
        # for i, r_ativo in enumerate(r_ativos):
        #     if y.ativo_equipe[r_ativo] == prob_def.n_equipes - 1:
        #         y.ativo_equipe[r_ativo] = 0
        #     else:
        #         y.ativo_equipe[r_ativo] += 1
        #     if i >= 30:
        #         break

    # trocando ativo de equipe e equipe de base - 1.953.125 vizinhos
    elif k == 4:
        for equipe, base in enumerate(y.equipe_base):
            y.equipe_base[equipe] = np.random.randint(prob_def.n_bases)
        for i, r_ativo in enumerate(r_ativos):
            if y.ativo_equipe[r_ativo] == prob_def.n_equipes - 1:
                y.ativo_equipe[r_ativo] = 0
            else:
                y.ativo_equipe[r_ativo] += 1
            if i >= 30:
                break
    return y


def first_improvement(x: solution, k: int, objective_function: callable, prob_def, max_iteration=2e6):
    current_fitness = objective_function(x, prob_def).fitness_penalizado
    neighbor_fitness = np.inf
    it = 0
    max_neighbors = {
        1:3,
        2:125,
        3:50, #1.9e6
        4:50
    }
    while neighbor_fitness > current_fitness and it < max_iteration:
        neighbor = shake(x, k, prob_def)
        neighbor_fitness = objective_function(neighbor, prob_def).fitness_penalizado
        it += 1
        if it > max_neighbors[k]:
            neighbor = x
            break
    return neighbor


'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    if y.fitness_penalizado < x.fitness_penalizado:
        x = deepcopy(y)
        k = 1
    else:
        k += 1

    return x, k

def RVNS(prob_def, initial_solution, objective_function, max_iteration, historico, kmax=3):
    it = 0
    current_solution = initial_solution
    # Ciclo iterativo do método
    while it < max_iteration:
        k = 1
        while k <= kmax:
            # Gera uma solução candidata na k-ésima vizinhança de x
            new_solution = shake(current_solution, k, prob_def)
            new_solution = objective_function(new_solution, prob_def)
            it += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            current_solution, k = neighborhoodChange(current_solution, new_solution, k)

            # Armazena dados para plot
            historico.update(current_solution)
    return historico

def BasicVNS(prob_def, initial_solution, objective_function, max_iteration, historico, kmax=4):
    it = 0
    current_solution = initial_solution
    # Ciclo iterativo do método
    for it in tqdm(range(max_iteration)):
        k = 1
        while k <= kmax:
            # Gera uma solução candidata na k-ésima vizinhança de x
            new_solution = first_improvement(
                x=current_solution,
                k=k,
                objective_function=objective_function,
                prob_def=prob_def
            )
            new_solution = objective_function(new_solution, prob_def)
            # it += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            current_solution, k = neighborhoodChange(current_solution, new_solution, k)

            # Armazena dados para plot
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

    prob_def = problem_definition(
        base_map=bases_map,
        ativo_map=ativos_map,
        n_bases=len(bases_map),
        n_ativos=len(ativos_map),  # Número de ativos,
        n_equipes=3,
        distance_matrix=distance_matrix
    )
    return prob_def


def optimize(fobj, apply_constructive_heuristic=False, max_it=40e3, follow_optimizitation=False):
    historicos = []
    for _ in range(5):
        # Contador do número de soluções candidatas avaliadas
        num_sol_avaliadas = 0

        # Máximo número de soluções candidatas avaliadas
        max_num_sol_avaliadas = max_it

        # Faz a leitura dos dados da instância do problema
        prob_def = get_problem_definition()

        # Gera solução inicial
        x = sol_inicial(prob_def, apply_constructive_heuristic=apply_constructive_heuristic)

        # Avalia solução inicial
        x = fobj(x, prob_def)
        num_sol_avaliadas += 1

        # Armazena dados para plot
        historico = history(min_iterations=max_it)
        historico.update(x)

        historico = BasicVNS(
            prob_def=prob_def,
            initial_solution=x,
            objective_function=fobj,
            max_iteration=max_num_sol_avaliadas,
            historico=historico
        )
        historicos.append(historico)
        if follow_optimizitation:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            s = len(historico.fit_pen)
            ax1.plot(np.linspace(0, s - 1, s), historico.fit_pen, 'k-')
            ax2.plot(np.linspace(0, s - 1, s), historico.pen, 'b:')
            fig.suptitle('Evolução da qualidade da solução candidata')
            ax1.set_ylabel('fitness(x) penalizado')
            ax2.set_ylabel('penalidade(x)')
            ax2.set_xlabel('Número de avaliações')
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
            plt.show()
    return historicos

def multiobjective_weighted(prob_def, max_iteration, follow_optimizitation):
    ws = WeightedSum()
    while True:
        # Armazena dados para plot
        historico = history(min_iterations=200)
        historico.update(deepcopy(initial_solution))
        historico = BasicVNS(
            prob_def=prob_def,
            initial_solution=deepcopy(initial_solution),
            objective_function=ws.weighted_sum,
            max_iteration=max_iteration,
            historico=historico
        )
        if follow_optimizitation:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            s = len(historico.fit_pen)
            ax1.plot(np.linspace(0, s - 1, s), historico.fit_pen, 'k-')
            ax2.plot(np.linspace(0, s - 1, s), historico.pen, 'b:')
            fig.suptitle('Evolução da qualidade da solução candidata')
            ax1.set_ylabel('fitness(x) penalizado')
            ax2.set_ylabel('penalidade(x)')
            ax2.set_xlabel('Número de avaliações')
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
            plt.show()
        ws.histories.append(historico)
        try:
            ws.balance()
        except:
            break
    return ws

if __name__=="__main__":

    historicos = []
    for _ in range(3):
        # Contador do número de soluções candidatas avaliadas
        num_sol_avaliadas = 0

        # Máximo número de soluções candidatas avaliadas
        max_num_sol_avaliadas = 100

        # Número de estruturas de vizinhanças definidas
        kmax = 3

        # Faz a leitura dos dados da instância do problema
        prob_def = get_problem_definition()

        # Gera solução inicial
        x = sol_inicial(prob_def, apply_constructive_heuristic=False, use_random=False)

        # Avalia solução inicial
        x = equilibrio_ativos(x, prob_def)
        num_sol_avaliadas += 1

        # Armazena dados para plot
        historico = history(min_iterations=200)
        historico.update(x)

        # historico = BasicVNS(
        #     prob_def=prob_def,
        #     initial_solution=x,
        #     objective_function=minimiza_distancia_maxima,
        #     max_iteration=max_num_sol_avaliadas,
        #     historico=historico
        # )
        historico = multiobjective_weighted(
            prob_def=prob_def,
            initial_solution=x,
            max_iteration=max_num_sol_avaliadas,
            follow_optimizitation=True
        )
        df = pd.DataFrame([h.best_solution.multi_fitness for h in historico.histories])
        print(df)
        # historicos.append(historico)
