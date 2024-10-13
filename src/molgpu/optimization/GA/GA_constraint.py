import numpy as np
from molgpu.optimization.GA.utils import (
    set_seeds,
    get_data,
    get_previous_data,
    usage_checker,
    get_fitness, normalize_data, matrix_relaxation
)
from molgpu.utils.utils import selected_enumeration
from pathlib import Path


class constraint_genetic_algorithm:
    def __init__(
            self,
            project_dir: Path,
            iteration: int,
            num_candidates: int,
            d: int = 20,  # resolution for random mutation
            max_num_genes_allowed: int = 4,
            top_k_fraction: float = 0.1,
            composition_mutation_rate: float = 0.1,
            composition_noise: float = 0.2,  # composition mutation noise
            gen_switch_mutation_rate: float = 0.1
    ):
        '''
        Genetic Algorithm for low-dimensional (d=12) composition optimization
            input_path: path to the input file
            fitness_path: path to the fitness file
            num_candidates: number of candidates in each generation including top-k parents and offspring
            top_k_fraction: fraction of the population that will be selected as parents
            d: number of genes in each chromosome, related to the accuracy of the composition
            max_num_genes_allowed: maximum number of non-zero genes allowed in each chromosome
            composition_mutation_rate: mutation rate for composition mutation
            composition_noise: noise added on composition mutation
            gen_switch_mutation_rate: mutation rate for gene switch mutation
        '''

        self.project_dir = project_dir
        self.iteration = iteration
        self.num_candidates = num_candidates
        self.max_num_genes_allowed = max_num_genes_allowed
        self.top_k_fraction = top_k_fraction
        self.composition_noise = composition_noise
        self.composition_mutation_rate = composition_mutation_rate
        self.gen_switch_mutation_rate = gen_switch_mutation_rate
        self.previous_candidate, self.previous_fitness = get_previous_data(project_dir)
        self.enumerated_candidates = selected_enumeration(4, 12, d)

    def selection(
            self,
            fitness: np.ndarray,
            k: int
    ) -> np.ndarray:
        '''
        pick top k parents
            fitness: 1 x d array
            k: number of parents to be chosen
        '''
        parents = np.argsort(fitness)[-k:]
        return parents

    def crossover(
            self,
            parents: np.ndarray,
            offspring_size: int
    ) -> np.ndarray:
        '''
        crossover the parents to generate offspring
            parents: parents
            offspring_size: size of offspring
        '''
        parents = normalize_data(parents)
        offspring = np.zeros((offspring_size, parents.shape[1]))
        for k in range(offspring_size):
            # randomly select 2 parents
            parent_indices = np.random.choice(
                parents.shape[0],
                size=2,
                replace=False
            )
            parent1_idx, parent2_idx = parent_indices
            # randomly decide the number of genes from parent1 
            num_genes_parent1 = np.random.randint(1, self.max_num_genes_allowed)
            # randomly select genes to pass to offspring based on the frequence
            genes_parent1 = np.random.choice(
                np.arange(parents.shape[1]),
                p=parents[parent1_idx],
                size=num_genes_parent1
            )
            genes_parent2 = np.random.choice(
                np.arange(parents.shape[1]),
                p=parents[parent2_idx],
                size=self.max_num_genes_allowed - num_genes_parent1
            )
            # pick the genes from parent1 and parent2 and put them into offspring
            # if there are same gene from parent1 and parent2, average them
            # pick the genes from parent1 and parent2 and put them into offspring
            offspring[k, genes_parent1] = parents[parent1_idx, genes_parent1]
            offspring[k, genes_parent2] = parents[parent2_idx, genes_parent2]
            common_genes = np.intersect1d(genes_parent1, genes_parent2)
            for gene in common_genes:
                offspring[k, gene] = (parents[parent1_idx, gene] + parents[parent2_idx, gene]) / 2
        offspring = normalize_data(offspring.round(2))
        if np.isnan(offspring).any():
            print('crossover')
        return offspring.round(2)

    def composition_mutation(
            self, 
            offspring: np.ndarray
    ) -> np.ndarray:
        '''
            randomly add gaussian noise to the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            # pick the non-zero genes
            non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
            for non_zero_gene in non_zero_genes:
                if np.random.uniform(0, 1) < self.composition_mutation_rate:
                    # randomly add a gaussian noise to the gene
                    mutated_offspring[idx, non_zero_gene] = mutated_offspring[idx, non_zero_gene] + np.random.normal(0, self.composition_noise)
                    if mutated_offspring[idx, non_zero_gene] < 0:
                        mutated_offspring[idx, non_zero_gene] = 0.01 
        if np.isnan(mutated_offspring).any():
            print('composition')
        mutated_offspring = normalize_data(mutated_offspring.round(2))
        
        return mutated_offspring.round(2)
    
    def gen_switch_mutation(
            self,
            offspring: np.ndarray     
    ) -> np.ndarray:
        '''
            randomly switch the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            if np.random.uniform(0, 1) < self.gen_switch_mutation_rate:
                # randomly select a non-zero gene
                non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
                gene_to_switch = np.random.choice(non_zero_genes)
                # randomly select a gene to switch
                gene_to_switch_with = np.random.choice(np.arange(offspring.shape[1]))
                if gene_to_switch == gene_to_switch_with:
                    continue
                # switch the genes, add the gene_to_switch to gene_to_switch_with together
                mutated_offspring[idx, gene_to_switch_with] = mutated_offspring[idx, gene_to_switch] + mutated_offspring[idx, gene_to_switch_with]
                mutated_offspring[idx, gene_to_switch] = 0
        mutated_offspring = normalize_data(mutated_offspring.round(2))
        if np.isnan(mutated_offspring).any():
            print('gen_switch')
        return mutated_offspring.round(2)

    def random_mutation(
            self,
            total_elements, 
            chosen_elements
    ) -> np.ndarray:
        '''
            randomly pick a candidate from the selected enumeration choices based on the cosine similarity
        '''
        a = np.zeros(total_elements)
        random_vector = np.random.rand(chosen_elements)
        random_vector = random_vector / random_vector.sum()
        a[:chosen_elements] = random_vector.round(1)
        np.random.shuffle(a)
        a = a / a.sum()

        return a.round(2)

    def propose_new_candidate(self) -> np.ndarray:

        set_seeds(42)

        input_path = self.project_dir / 'data' / f'round_{self.iteration}' / 'proposed_composition.csv'
        fitness_path = self.project_dir / 'data' / f'round_{self.iteration}' / 'activity.csv'
        x = get_data(input_path)
        fitness = get_fitness(fitness_path)
        # 1/10 of the population will be selected as for top-k parents
        k = int((x.shape[0] - 8) * self.top_k_fraction)
        offspring_size = self.num_candidates - k
        offspring_size_direct_mutation = int(offspring_size * 0.5)
        offspring_size_crossover_mutation = offspring_size - offspring_size_direct_mutation
        # remove first 8 rows for control experiments
        fitness = fitness[8:]
        population = x[8:]
        # select top-k parents from parents pool
        selected_parents_idx = self.selection(fitness, k)
        # crossover and mutation
        crossover_offspring = self.crossover(population[selected_parents_idx], offspring_size_crossover_mutation)
        mutated_crossover_offspring = self.composition_mutation(crossover_offspring)
        mutated_crossover_offspring = self.gen_switch_mutation(mutated_crossover_offspring)
        # randomly select parents from selected_parents_idx offspring_size_direct_mutation times
        direct_mutation_offspring = population[np.random.choice(selected_parents_idx, offspring_size_direct_mutation)]
        # direct composition mutation of the selected parents
        direct_mutation_offspring = self.composition_mutation(direct_mutation_offspring)
        # combine the offspring from crossover and direct mutation
        mutated_offspring = np.vstack((mutated_crossover_offspring, direct_mutation_offspring))

        for idx in range(mutated_offspring.shape[0]):
            # combine the previous candidates to the candidates before the current candidate
            previous_data = np.concatenate((self.previous_candidate, mutated_offspring[:idx]), axis=0)
            # Check if the mutated_offspring is the same as the previous data
            while True:
                is_similar_to_any = False  # Initialize flag as False
                for previous in previous_data:
                    if (mutated_offspring[idx] == previous).all():  # Compare the offspring with each previous candidate
                        is_similar_to_any = True  # Set flag to True if similar
                        break  # Break the loop as soon as a similar component is found
                if is_similar_to_any:
                    mutated_offspring[idx] = self.random_mutation(x.shape[1], self.max_num_genes_allowed)
                    continue
                else:
                    break  # Break the while loop if the offspring is not similar to any previous candidates

        population_next = np.vstack((population[selected_parents_idx], mutated_offspring)).round(2)
        population_next = normalize_data(population_next)

        return population_next.round(2)


class constraint_genetic_algorithm_hd:

    def __init__(
            self,
            project_dir: Path,
            iteration: int,
            num_candidates: int,
            volume_each_well: float = 20.0,  # resolution for random mutation 
            max_num_genes_allowed: int = 6,
            top_k_fraction: float = 0.1,
            composition_mutation_rate: float = 0.1,
            composition_noise: float = 0.2,  # noise added on composition mutation
            gen_switch_mutation_rate: float = 0.1
    ):
        '''
        Genetic Algorithm for high-dimensional (d=96, or 192) composition optimization
            self.project_dir = project_dir
            self.iteration = iteration
            self.num_candidates = num_candidates
            self.max_num_genes_allowed = max_num_genes_allowed
            self.top_k_fraction = top_k_fraction
            self.composition_noise = composition_noise
            self.composition_mutation_rate = composition_mutation_rate
            self.gen_switch_mutation_rate = gen_switch_mutation_rate
            self.previous_candidate, self.previous_fitness = get_previous_data(project_dir)
            self.enumerated_candidates = selected_enumeration(4, 12, d)
        '''
        self.project_dir = project_dir
        self.iteration = iteration
        self.num_candidates = num_candidates
        self.max_num_genes_allowed = max_num_genes_allowed
        self.top_k_fraction = top_k_fraction
        self.composition_noise = composition_noise
        self.composition_mutation_rate = composition_mutation_rate
        self.gen_switch_mutation_rate = gen_switch_mutation_rate
        self.previous_candidate, self.previous_fitness = get_previous_data(project_dir)
        self.volume_each_well = volume_each_well
        
    def selection(
            self,
            fitness: np.ndarray,
            k: int
    ) -> np.ndarray:
        '''
        choose top k parents
            fitness: 1 x d array
            k: number of parents to be chosen
        '''
        parents = np.argsort(fitness)[-k:]
        return parents

    def crossover(
            self,
            parents: np.ndarray,
            offspring_size: int
    ) -> np.ndarray:
        '''
        select 2 parents and randomly select genes from each parent for crossover
            parents: parents
            offspring_size: size of offspring
        '''
        parents = normalize_data(parents)
        offspring = np.zeros((offspring_size, parents.shape[1]))

        for k in range(offspring_size):
            # randomly select 2 parents
            parent_indices = np.random.choice(parents.shape[0], size=2, replace=False)
            parent1_idx, parent2_idx = parent_indices
            # randomly decide the number of genes from parent1
            num_genes_parent1 = np.random.randint(1, self.max_num_genes_allowed)
            # randomly select genes to pass to offspring based on the frequence
            genes_parent1 = np.random.choice(
                np.arange(parents.shape[1]),
                p=parents[parent1_idx],
                size=num_genes_parent1
            )
            genes_parent2 = np.random.choice(
                np.arange(parents.shape[1]),
                p=parents[parent2_idx],
                size=self.max_num_genes_allowed - num_genes_parent1
            )
            # pick the genes from parent1 and parent2 and put them into offspring
            # if there are same gene from parent1 and parent2, average them
            # pick the genes from parent1 and parent2 and put them into offspring
            offspring[k, genes_parent1] = parents[parent1_idx, genes_parent1]
            offspring[k, genes_parent2] = parents[parent2_idx, genes_parent2]
            common_genes = np.intersect1d(genes_parent1, genes_parent2)
            for gene in common_genes:
                offspring[k, gene] = (parents[parent1_idx, gene] + parents[parent2_idx, gene]) / 2
        offspring = normalize_data(offspring.round(2))
        if np.isnan(offspring).any():
            print('crossover')
        return offspring.round(2)

    def composition_mutation(
            self,
            offspring: np.ndarray
    ) -> np.ndarray:
        '''
            randomly add gaussian noise to the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            # pick the non-zero genes
            non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
            for non_zero_gene in non_zero_genes:
                if np.random.uniform(0, 1) < self.composition_mutation_rate:
                    # randomly add a gaussian noise to the gene
                    mutated_offspring[idx, non_zero_gene] = mutated_offspring[idx, non_zero_gene] + np.random.normal(0, self.composition_noise)
                    if mutated_offspring[idx, non_zero_gene] < 0:
                        mutated_offspring[idx, non_zero_gene] = 0.01
        if np.isnan(mutated_offspring).any():
            print('composition')

        mutated_offspring = normalize_data(mutated_offspring.round(2))

        return mutated_offspring.round(2)
    
    def gen_switch_mutation(
            self,
            offspring: np.ndarray
    ) -> np.ndarray:
        '''
        randomly switch the genes
        '''
        mutated_offspring = offspring
        for idx in range(mutated_offspring.shape[0]):
            if np.random.uniform(0, 1) < self.gen_switch_mutation_rate:
                # randomly select a non-zero gene
                non_zero_genes = np.nonzero(mutated_offspring[idx])[0]
                gene_to_switch = np.random.choice(non_zero_genes)
                # randomly select a gene to switch
                gene_to_switch_with = np.random.choice(np.arange(offspring.shape[1]))
                if gene_to_switch == gene_to_switch_with:
                    continue
                # switch the genes, add the gene_to_switch to gene_to_switch_with together
                mutated_offspring[idx, gene_to_switch_with] = mutated_offspring[idx, gene_to_switch] + mutated_offspring[idx, gene_to_switch_with]
                mutated_offspring[idx, gene_to_switch] = 0

        mutated_offspring = normalize_data(mutated_offspring.round(2))
        if np.isnan(mutated_offspring).any():
            print('gen_switch')

        return mutated_offspring.round(2)

    def random_generation(
            self,
            total_elements,
            chosen_elements):
        '''
        total_elements: total number of elements in the array
        chosen_elements: number of elements to be chosen
        '''
        a = np.zeros(total_elements)
        random_vector = np.random.rand(chosen_elements)
        random_vector = random_vector / random_vector.sum()
        a[:chosen_elements] = random_vector.round(1)
        np.random.shuffle(a)
        a = a / a.sum()
        return a.round(1)

    def propose_new_candidate(self) -> np.ndarray:

        set_seeds(42)
        input_path = self.project_dir / 'data' / f'round_{self.iteration}' / 'proposed_composition.csv'
        fitness_path = self.project_dir / 'data' / f'round_{self.iteration}' / 'activity.csv'
        x = get_data(input_path)
        fitness = get_fitness(fitness_path)
        # 1/10 of the population will be selected as for top-k parents
        k = int((x.shape[0] - 8) * self.top_k_fraction)
        offspring_size = self.num_candidates - k
        offspring_size_direct_mutation = int(offspring_size * 0.5)
        offspring_size_crossover_mutation = offspring_size - offspring_size_direct_mutation
        # remove first 8 rows for control experiments
        fitness = fitness[8:]
        population = x[8:]
        # select top-k parents from parents pool
        selected_parents_idx = self.selection(fitness, k)
        # crossover and mutation
        crossover_offspring = self.crossover(population[selected_parents_idx], offspring_size_crossover_mutation)
        mutated_crossover_offspring = self.composition_mutation(crossover_offspring)
        mutated_crossover_offspring = self.gen_switch_mutation(mutated_crossover_offspring)
        # randomly select parents from selected_parents_idx offspring_size_direct_mutation times
        direct_mutation_offspring = population[np.random.choice(selected_parents_idx, offspring_size_direct_mutation)]
        # direct composition mutation of the selected parents
        direct_mutation_offspring = self.composition_mutation(direct_mutation_offspring)
        # combine the offspring from crossover and direct mutation
        mutated_offspring = np.vstack((mutated_crossover_offspring, direct_mutation_offspring))
        mutated_offspring = matrix_relaxation(mutated_offspring)

        previous_data = self.previous_candidate
        for idx in range(mutated_offspring.shape[0]):
            # Check if the mutated_offspring is the same as the previous data
            while True:
                need_to_change = False  # Initialize flag as False
                for previous in previous_data:
                    if (mutated_offspring[idx] == previous).all() or usage_checker(mutated_offspring[idx], previous_data, self.volume_each_well):
                        need_to_change = True  # Set flag to True if similar
                        break  # Break the loop as soon as a similar component is found
                if need_to_change:
                    mutated_offspring[idx] = self.random_generation(x.shape[1], self.max_num_genes_allowed)
                    mutated_offspring[idx] = matrix_relaxation(mutated_offspring[idx])
                    continue
                else:
                    break  # Break the while loop if the offspring is not similar to any previous candidates
            previous_data = np.vstack((previous_data, mutated_offspring[idx]))
        population_next = np.vstack(
            (population[selected_parents_idx], mutated_offspring)
        )
        population_next = matrix_relaxation(population_next)

        return population_next.round(1)
