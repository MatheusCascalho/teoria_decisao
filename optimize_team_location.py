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
np.random.seed(11)
prob_falha = np.abs(np.random.normal(size=125))

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
    prob_falha: list = field(default_factory=list)

    def __post_init__(self):
        if len(self.prob_falha)==0:
            total = self.distance_matrix.sum().sum()
            max_distance = 1
            self.prob_falha = prob_falha

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

def minimiza_distancia_para_falha(x: solution, prob_def: problem_definition):
    distancias = []
    for ativo, base in enumerate(x.ativo_base):
        d = prob_def.distance_matrix.loc[ativo, base] * prob_def.prob_falha[ativo]
        distancias.append(d)

    x.fitness = sum(distancias)
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
        f2 = minimiza_distancia_para_falha(new_x, prob_def).fitness_penalizado
        min_f2 = prob_def.distance_matrix.min().min()
        max_f2 = prob_def.distance_matrix.max().max()
        min_f1 = sum(sorted(prob_def.distance_matrix.values.reshape((125*14)))[:prob_def.n_ativos])
        max_f1 = sum(np.abs(sorted(prob_def.distance_matrix.values.reshape((125*14))*-1))[:prob_def.n_ativos])

        f2_norm = (f2 - min_f1) / (max_f1 - min_f1)
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
        
class EpsilonConstraint:
    def __init__(self, epsilon_values, objective_index=0):
        """
        Classe para aplicar o método ε-restrito.

        Parameters:
            epsilon_values (list): Lista de valores ε para restringir os objetivos secundários.
            objective_index (int): Índice do objetivo a ser minimizado diretamente.
        """
        self.epsilon_values = epsilon_values
        self.objective_index = objective_index
        self.epsilon = self.epsilon_values[0]

    def epsilon_constrained_function(self, x, prob_def, objectives):
        """
        Função objetivo para ε-restrito com penalidades para violações de restrições.

        Parameters:
            x (solution): Solução a ser avaliada.
            prob_def (problem_definition): Definição do problema.
            objectives (list): Lista de funções objetivo.

        Returns:
            solution: Solução avaliada com penalidades aplicadas.
        """
        primary_objective = objectives[self.objective_index](deepcopy(x), prob_def).fitness
        secund_objective = objectives[self.objective_index+1](deepcopy(x), prob_def).fitness - self.epsilon
        # penalties = 0
        u = 100

        secund_objective = u * max(0, secund_objective)**2

        #  u * max(

        # for i, obj_fn in enumerate(objectives):
        #     if i != self.objective_index:
        #         value = obj_fn(deepcopy(x), prob_def).fitness
        #         if value > self.epsilon_values[i - 1]:
        #             penalties += (value - self.epsilon_values[i - 1]) ** 2

        x.fitness = primary_objective
        x.penalidade = secund_objective

        return x


def run_epsilon_restricted(prob_def, objectives, epsilon_ranges, follow_optimizitation, max_iteration=2000, tests=5):
    """
    Executa a abordagem ε-restrito para gerar soluções multiobjetivo.

    Parameters:
        prob_def (problem_definition): Definição do problema.
        objectives (list): Lista de funções objetivo.
        epsilon_ranges (list): Intervalos de ε para restringir os objetivos secundários.
        max_iteration (int): Número máximo de iterações.
        tests (int): Número de testes a serem realizados.

    Returns:
        list: Lista de fronteiras Pareto geradas em cada execução.
    """
    pareto_fronts = []

    for _ in range(tests):
        epsilon_values = np.linspace(epsilon_ranges[0], epsilon_ranges[1], 350)

        solutions = []
        epsilon_constraint = EpsilonConstraint(epsilon_values, objective_index=0)

        for epsilon in epsilon_values:
            epsilon_constraint.epsilon = epsilon
            initial_solution = sol_inicial(prob_def)
            initial_solution = epsilon_constraint.epsilon_constrained_function(initial_solution, prob_def, objectives)

            historico = history(min_iterations=max_iteration)
            historico.update(initial_solution)

            historico = BasicVNS(
                prob_def=prob_def,
                initial_solution=initial_solution,
                objective_function=lambda x, p: epsilon_constraint.epsilon_constrained_function(x, p, objectives),
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

            solutions.append(deepcopy(historico.best_solution))

        pareto_fronts.append(solutions)

    return pareto_fronts

def plot_pareto_fronts(pareto_fronts, title="Fronteiras Pareto - Método ε-restrito"):
    """
    Plota as fronteiras Pareto estimadas.

    Parameters:
        pareto_fronts (list): Fronteiras Pareto geradas.
        title (str): Título do gráfico.
    """
    plt.figure(figsize=(10, 6))
    for front in pareto_fronts:
        f1 = [solution.multi_fitness['f1'] for solution in front]
        f2 = [solution.multi_fitness['f2'] for solution in front]
        plt.scatter(f1, f2, label="Fronteira Pareto")
    plt.title(title)
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.legend()
    plt.grid()
    plt.show()


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
    for it in tqdm(range(max_iteration), desc='Refinamento BVNS'):
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

    prob_falha = pd.read_excel('probfalhaativos.xlsx')

    prob_def = problem_definition(
        base_map=bases_map,
        ativo_map=ativos_map,
        n_bases=len(bases_map),
        n_ativos=len(ativos_map),  # Número de ativos,
        n_equipes=3,
        distance_matrix=distance_matrix,
        prob_falha=prob_falha['prob'].values
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
        # Gera solução inicial
        initial_solution = sol_inicial(prob_def, apply_constructive_heuristic=False, use_random=False)

        # Avalia solução inicial
        initial_solution = ws.weighted_sum(initial_solution, prob_def)

        # Armazena dados para plot
        historico = history(min_iterations=max_iteration)
        historico.update(initial_solution)
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

def get_ws_multiobjective(max_num_sol_avaliadas=100, tests=5):
    borders = []
    for _ in tqdm(range(tests), desc=f'Teste - construção de fronteiras'):
        # Faz a leitura dos dados da instância do problema
        prob_def = get_problem_definition()

        border = multiobjective_weighted(
            prob_def=prob_def,
            max_iteration=max_num_sol_avaliadas,
            follow_optimizitation=True
        )
        borders.append(border)
    return borders

def is_dominated(sol_a, sol_b):
    """Verifica se sol_a é dominada por sol_b."""
    is_not_worse_in_all = all(a >= b for a, b in zip(sol_a, sol_b))
    is_strictly_worse_in_at_least_one = any(a > b for a, b in zip(sol_a, sol_b))
    return is_not_worse_in_all and is_strictly_worse_in_at_least_one


def find_non_dominated_solutions(df):
    """Identifica as soluções não dominadas."""
    non_dominated = []
    for i, sol_a in df.iterrows():
        dominated = False
        for j, sol_b in df.iterrows():
            if i != j and is_dominated(sol_a, sol_b):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return df.iloc[non_dominated]

# # Identificar as soluções não dominadas
# non_dominated_solutions = find_non_dominated_solutions(df_unique)

#if __name__=='__main__':
    #fronteiras = get_ws_multiobjective(max_num_sol_avaliadas=int(2e3), tests=5)
    #import pickle

    # Salvando em um arquivo pickle
    #with open("fronteiras_prob_falha.pkl", "wb") as pickle_file:
    #    pickle.dump(fronteiras, pickle_file)

    #print("Dados salvos em data.pkl")

    # historicos_f1 = optimize(minimiza_distancia_para_falha, max_it=500, follow_optimizitation=True)
    # for historico in historicos_f1:
    #     plt.plot(historico.fit)

if __name__ == '__main__':
    # Soma Ponderada
    fronteiras_ws = get_ws_multiobjective(max_num_sol_avaliadas=int(2), tests=5)
    import pickle

    # Salvando resultados da soma ponderada
    with open("fronteiras_ws.pkl", "wb") as pickle_file:
        pickle.dump(fronteiras_ws, pickle_file)
    print("Fronteiras da soma ponderada salvas em fronteiras_ws.pkl")
    
    # Epsilon-restrito
    prob_def = get_problem_definition()
    objectives = [minimiza_distancias, minimiza_distancia_para_falha]
    epsilon_ranges = [300, 7000]  # Intervalos de ε para os objetivos
    
    fronteiras_eps = run_epsilon_restricted(
        prob_def=prob_def,
        objectives=objectives,
        epsilon_ranges=epsilon_ranges,
        follow_optimizitation=True,
        max_iteration=2,
        tests=5
    )
    
    # Salvando resultados do epsilon-restrito
    with open("fronteiras_eps.pkl", "wb") as pickle_file:
        pickle.dump(fronteiras_eps, pickle_file)
    print("Fronteiras do ε-restrito salvas em fronteiras_eps.pkl")
    
    # Visualizar ambas as fronteiras
    def plot_fronteiras_duplas(fronteiras_ws, fronteiras_eps):
        plt.figure(figsize=(12, 8))
        
        # Fronteiras da soma ponderada
        for front in fronteiras_ws:
            f1 = [solution.multi_fitness['f1'] for solution in front]
            f2 = [solution.multi_fitness['f2'] for solution in front]
            plt.scatter(f1, f2, label="Soma Ponderada", alpha=0.7, color='blue')
        
        # Fronteiras do epsilon-restrito
        for front in fronteiras_eps:
            f1 = [solution.multi_fitness['f1'] for solution in front]
            f2 = [solution.multi_fitness['f2'] for solution in front]
            plt.scatter(f1, f2, label="ε-restrito", alpha=0.7, color='red')
        
        plt.title("Fronteiras Pareto Estimadas")
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Chamar a função de plotagem
    plot_fronteiras_duplas(fronteiras_ws, fronteiras_eps)