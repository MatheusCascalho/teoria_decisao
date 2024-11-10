import pandas as pd
import seaborn as sns

'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import copy

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
    fitness_penalizado: float = 0

    @property
    def ativo_base(self):
        x = []
        for equipe in self.ativo_equipe:
            base = self.equipe_base[equipe]
            x.append(base)
        return x


@dataclass
class history:
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
        return n_sol>=200 and all(s.fitness_penalizado == self.sol[-1].fitness_penalizado for s in self.sol[-min_stabel:])

'''
Implementa uma solução inicial para o problema
'''
def sol_inicial(prob_def: problem_definition, apply_constructive_heuristic=False):
    '''
    Modelou-se uma solução x como um vetor binário

    x = [x1 x2 ... xn]
    '''

    if apply_constructive_heuristic == False:
        # Constrói solução inicial aleatoriamente
        y = np.random.randint(0, prob_def.n_bases, size=prob_def.n_equipes)
        h = np.random.randint(0, prob_def.n_equipes, size=prob_def.n_ativos)
        sol = solution(equipe_base=y, ativo_equipe=h)
    else:  # TODO
        pass
    return sol


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

    # print(total_distance)
    ativos_por_equipe = [len(np.where(np.array(x.ativo_equipe) == k)[0]) for k in range(prob_def.n_equipes)]
    penalidade = 0
    min_ativos = prob_def.ETA * prob_def.n_ativos / prob_def.n_equipes
    # print(f"Minimo de ativos:{min_ativos}")
    u = 100
    for equipe, qtd_ativos in enumerate(ativos_por_equipe):
        g = min_ativos - qtd_ativos
        # print(f"Qtd. Ativos: {qtd_ativos} \t G: {g}")
        penalidade += u * max(0, g) ** 2

    x.penalidade = penalidade
    x.fitness_penalizado = total_distance + penalidade
    # print(ativos_por_equipe)

    return x


'''
Implementa a função shake
'''
def shake(x: solution, k: int, prob_def: problem_definition):
    y = copy.deepcopy(x)
    r_equipe = np.random.randint(prob_def.n_equipes)
    r_ativo = np.random.randint(prob_def.n_ativos)

    # trocando equipe de base - 3 vizinhos
    if k == 1:
        # y.solution[r[0]] = not(y.solution[r[0]])
        if y.equipe_base[r_equipe] == prob_def.n_bases - 1:
            y.equipe_base[r_equipe] = 0
        else:
            y.equipe_base[r_equipe] += 1

    # trocando ativo de equipe - 125 vizinhos
    elif k == 2:
        if y.ativo_equipe[r_ativo] == prob_def.n_equipes - 1:
            y.ativo_equipe[r_ativo] = 0
        else:
            y.ativo_equipe[r_ativo] += 1

    # trocando ativo de equipe e equipe de base - 1.953.125 vizinhos
    elif k == 3:
        if y.equipe_base[r_equipe] == prob_def.n_bases - 1:
            y.equipe_base[r_equipe] = 0
        else:
            y.equipe_base[r_equipe] += 1
        if y.ativo_equipe[r_ativo] == prob_def.n_equipes - 1:
            y.ativo_equipe[r_ativo] = 0
        else:
            y.ativo_equipe[r_ativo] += 1

    return y


def first_improvement(x: solution, k: int, objective_function: callable, prob_def, max_iteration=2e6):
    current_fitness = objective_function(x, prob_def).fitness_penalizado
    neighbor_fitness = np.inf
    it = 0
    max_neighbors = {
        1:3,
        2:125,
        3:20#1.9e6
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
        x = copy.deepcopy(y)
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

def BasicVNS(prob_def, initial_solution, objective_function, max_iteration, historico, kmax=3):
    it = 0
    current_solution = initial_solution
    # Ciclo iterativo do método
    while it <= max_iteration:
        k = 1
        while k < kmax:
            # Gera uma solução candidata na k-ésima vizinhança de x
            new_solution = first_improvement(
                x=current_solution,
                k=k,
                objective_function=objective_function,
                prob_def=prob_def
            )
            new_solution = objective_function(new_solution, prob_def)
            it += 1

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


if __name__=="__main__":

    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = 10000

    # Número de estruturas de vizinhanças definidas
    kmax = 3

    # Faz a leitura dos dados da instância do problema
    prob_def = get_problem_definition()

    # Gera solução inicial
    x = sol_inicial(prob_def, apply_constructive_heuristic=False)

    # Avalia solução inicial
    x = minimiza_distancias(x, prob_def)
    num_sol_avaliadas += 1

    # Armazena dados para plot
    historico = history()
    historico.update(x)

    historico = RVNS(
        initial_solution=x,
        objective_function=minimiza_distancias,
        max_iteration=max_num_sol_avaliadas,
        historico=historico
    )

    # print('\n--- SOLUÇÃO INICIAL CONSTRUÍDA ---\n')
    # print('Identificação dos projetos selecionados:\n')
    # print('x = {}\n'.format(historico.sol[0]))
    # print('fitness(x) = {:.2f}\n'.format(historico.fit[0]))
    # print('fitness(x) penalizado = {:.2f}\n'.format(historico.fit_pen[0]))
    # print('penalidade(x) = {:.2f}\n'.format(historico.pen[0]))

    # print('\n--- MELHOR SOLUÇÃO ENCONTRADA ---\n')
    # print('Identificação dos projetos selecionados:\n')
    # print('x = {}\n'.format(x.ativo_base))
    # print('fitness(x) = {:.2f}\n'.format(x.fitness))
    # print('fitness(x) penalizado = {:.2f}\n'.format(x.fitness_penalizado))
    # print('penalidade(x) = {:.2f}\n'.format(x.penalidade))

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
