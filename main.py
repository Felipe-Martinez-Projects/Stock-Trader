import numpy as np
import sys


class Chromosome:
    def __init__(self, lower_bound_1=None, upper_bound_1=None, lower_bound_2=None, upper_bound_2=None, action=None):
        if all(vars is None for vars in [lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2, action]):
            # Generate a random chromosome

            genes = np.random.normal(0, 1.15, 4)
            #print(genes)
            # Swap
            if genes[0] > genes[1]:
                genes[0], genes[1] = genes[1], genes[0]

            if genes[2] > genes[3]:
                genes[2], genes[3] = genes[3], genes[2]

            action = np.random.choice([0, 1])
            lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2 = genes

            #print("Genes: ", genes)
            #print("Action: ", action)

        # Check that the ranges are valid
        assert lower_bound_1 <= upper_bound_1, "First range is invalid!"
        assert lower_bound_2 <= upper_bound_2, "Second range is invalid!"

        # Store the ranges and action
        self.lower_bound_1 = lower_bound_1
        self.upper_bound_1 = upper_bound_1
        self.lower_bound_2 = lower_bound_2
        self.upper_bound_2 = upper_bound_2
        self.action = action

    def matches(self, day1_change, day2_change):
        # Check if the chromosome matches the provided data line
        return (self.lower_bound_1 <= day1_change <= self.upper_bound_1 and
                self.lower_bound_2 <= day2_change <= self.upper_bound_2)

    def compute_fitness(self, data):
        # Compute the fitness of the chromosome against the data
        total_fitness = 0
        matches_count = 0

        for change_day1, change_day2, profit in data:
            if self.matches(change_day1, change_day2):
                matches_count += 1

                #action is buy
                if self.action == 1:
                    total_fitness += profit

                #action is short
                else:
                    total_fitness -= profit

        #if no matches found, return -5000
        if matches_count == 0:
            return -5000
        #return the fitness
        return total_fitness



# Load training data from the given file
def load_training_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.split())) for line in lines]
    return data


# Test a chromosome against the training data
def test_chromosome(chromosome, training_data):
    matches = 0
    for data_line in training_data:
        if chromosome.matches(data_line[0], data_line[1]):
            matches += 1
    return matches

def initialize_population(population_size):
    # Initialize population of chromosomes
    return [Chromosome() for _ in range(population_size)]


def roulette_wheel_selection(population, fitnesses):
    # Fix Negative Fitnesses
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        fitnesses = [fitness - min_fitness for fitness in fitnesses]

    total_fitness = sum(fitnesses)
    # If total fitness is 0, return a random chromosome
    if total_fitness == 0:
        return np.random.choice(population)

    random_value = np.random.uniform(0, total_fitness)
    running_sum = 0

    for chromosome, fitness in zip(population, fitnesses):
        #print(fitness)
        running_sum += fitness
        if running_sum > random_value:
            return chromosome


def elitist_selection(population, fitnesses, num_to_select):
    # First pair chromosome with fitness
    population_fitness = list(zip(population, fitnesses))
    # Sort by fitness
    sorted_pairs = sorted(population_fitness, key=lambda x: x[1], reverse=True)
    # Get the top chromosomes
    selected_chromosomes = [pair[0] for pair in sorted_pairs[:num_to_select]]
    return selected_chromosomes


def select_roulette(population, fitnesses, num_to_select):
    selected = []
    for _ in range(num_to_select):
        selected.append(roulette_wheel_selection(population, fitnesses))
    return selected


def uniform_crossover(parent1, parent2):
    # Create a new chromosome using uniform crossover of 2 parents
    child1 = Chromosome(0, 0, 0, 0, 0)
    child2 = Chromosome(0, 0, 0, 0, 0)

    genes = ["lower_bound_1", "upper_bound_1", "lower_bound_2", "upper_bound_2", "action"]

    for gene in genes:
        if np.random.rand() < 0.5:
            setattr(child1, gene, getattr(parent1, gene))
            setattr(child2, gene, getattr(parent2, gene))
        else:
            setattr(child1, gene, getattr(parent2, gene))
            setattr(child2, gene, getattr(parent1, gene))

    # Check that the ranges are valid, swap if not
    if child1.lower_bound_1 > child1.upper_bound_1:
        child1.lower_bound_1, child1.upper_bound_1 = child1.upper_bound_1, child1.lower_bound_1

    if child1.lower_bound_2 > child1.upper_bound_2:
        child1.lower_bound_2, child1.upper_bound_2 = child1.upper_bound_2, child1.lower_bound_2

    if child2.lower_bound_1 > child2.upper_bound_1:
        child2.lower_bound_1, child2.upper_bound_1 = child2.upper_bound_1, child2.lower_bound_1

    if child2.lower_bound_2 > child2.upper_bound_2:
        child2.lower_bound_2, child2.upper_bound_2 = child2.upper_bound_2, child2.lower_bound_2

    #print("Parents")
    #for parent in [parent1, parent2]:
    #    print(vars(parent))

    #print("Child1")
    #print(vars(child1))
    #print("Child2")
    #print(vars(child2))

    return child1, child2


def k_point_crossover(parent1, parent2, k):
    assert 1 <= k <= 4, "Invalid number of crossover points!"

    # Get genes from parent 1
    genes1 = [parent1.lower_bound_1, parent1.upper_bound_1, parent1.lower_bound_2, parent1.upper_bound_2, parent1.action]
    # Get genes from parent 2
    genes2 = [parent2.lower_bound_1, parent2.upper_bound_1, parent2.lower_bound_2, parent2.upper_bound_2, parent2.action]

    crossover_points = sorted(np.random.choice(range(1, len(genes1)), k, replace=False))

    #print("Crossover points: ", crossover_points)

    # Create a new chromosome using k-point crossover of 2 parents
    child1 = Chromosome(0, 0, 0, 0, 0)
    child2 = Chromosome(0, 0, 0, 0, 0)
    child1_genes = []
    child2_genes = []

    parent1_turn = True
    for i in range(len(genes1)):
        if i in crossover_points:
            parent1_turn = not parent1_turn
        if parent1_turn:
            child1_genes.append(genes1[i])
            child2_genes.append(genes2[i])
        else:
            child1_genes.append(genes2[i])
            child2_genes.append(genes1[i])

    child1.lower_bound_1, child1.upper_bound_1, child1.lower_bound_2, child1.upper_bound_2, child1.action = child1_genes
    child2.lower_bound_1, child2.upper_bound_1, child2.lower_bound_2, child2.upper_bound_2, child2.action = child2_genes
    #print("Parents")
    #for parent in [parent1, parent2]:
    #    print(vars(parent))

    #print("Child")
    #print(vars(child))
    return child1, child2


def mutate_chromosome(chromosome, mutation_rate):
    # Iterate over each gene in the chromosome (5 genes)
    mutated_chromosome = Chromosome(0, 0, 0, 0, 0)
    genes = ["lower_bound_1", "upper_bound_1", "lower_bound_2", "upper_bound_2", "action"]

    # Gene 1 - Lower bound 1
    if np.random.rand() <= mutation_rate:
        #chromosome.lower_bound_1 = np.random.normal(0, 1.15)
        mutated_chromosome.lower_bound_1 = np.random.normal(0, 1.15)
    else:
        #print("HI")
        mutated_chromosome.lower_bound_1 = chromosome.lower_bound_1

    # Gene 2 - Upper bound 1
    if np.random.rand() <= mutation_rate:
        #chromosome.upper_bound_1 = np.random.normal(0, 1.15)
        mutated_chromosome.upper_bound_1 = np.random.normal(0, 1.15)
    else:
        mutated_chromosome.upper_bound_1 = chromosome.upper_bound_1

    # Gene 3 - Lower bound 2
    if np.random.rand() <= mutation_rate:
        #chromosome.lower_bound_2 = np.random.normal(0, 1.15)
        mutated_chromosome.lower_bound_2 = np.random.normal(0, 1.15)
    else:
        mutated_chromosome.lower_bound_2 = chromosome.lower_bound_2

    # Gene 4 - Upper bound 2
    if np.random.rand() <= mutation_rate:
        #chromosome.upper_bound_2 = np.random.normal(0, 1.15)
        mutated_chromosome.upper_bound_2 = np.random.normal(0, 1.15)
    else:
        mutated_chromosome.upper_bound_2 = chromosome.upper_bound_2

    # Gene 5 - Action
    if np.random.rand() <= mutation_rate:
        #chromosome.action = np.random.choice([0, 1])
        mutated_chromosome.action = np.random.choice([0, 1])
    else:
        mutated_chromosome.action = chromosome.action

    # Check that the ranges are valid, swap if not
    if mutated_chromosome.lower_bound_1 > mutated_chromosome.upper_bound_1:
        mutated_chromosome.lower_bound_1, mutated_chromosome.upper_bound_1 = mutated_chromosome.upper_bound_1, mutated_chromosome.lower_bound_1

    if mutated_chromosome.lower_bound_2 > mutated_chromosome.upper_bound_2:
        mutated_chromosome.lower_bound_2, mutated_chromosome.upper_bound_2 = mutated_chromosome.upper_bound_2, mutated_chromosome.lower_bound_2


    # Check that the ranges are valid, swap if not
    #if chromosome.lower_bound_1 > chromosome.upper_bound_1:
    #    chromosome.lower_bound_1, chromosome.upper_bound_1 = chromosome.upper_bound_1, chromosome.lower_bound_1

    # if chromosome.lower_bound_2 > chromosome.upper_bound_2:
    #     chromosome.lower_bound_2, chromosome.upper_bound_2 = chromosome.upper_bound_2, chromosome.lower_bound_2

    return mutated_chromosome

def mutate_population(population, mutation_rate):
    new_population = []

    for chromosome in population:
        new_chromosome = mutate_chromosome(chromosome, mutation_rate)
        new_population.append(new_chromosome)
        #mutate_chromosome(chromosome, mutation_rate)
        #new_population.append(chromosome)

    return new_population


def create_next_generation(population, fitnesses, x_percentage, selection, num_to_clone):
    # Select X% of the population to clone
    #population_size = len(population)
    #num_to_clone = int(population_size * x_percentage / 100)
    #roulette_num_to_clone = population_size - num_to_clone

    if num_to_clone % 2 != 0:
        num_to_clone += 1

    if num_to_clone < 2:
        print("Please be sure to have the number of chromosomes greater than 2 and percentage that will allow for atleast 2 parents")
        print("DEFAULTING TO 2 PARENTS")
        num_to_clone = 2

    if selection == "Roulette":
        #print("Roulette Selection")
        # Select X% of the population to clone
        cloned_chromosomes = select_roulette(population, fitnesses, num_to_clone)

    elif selection == "Elitist":
        #print("Elitist Selection")
        # Select X% of the population to clone
        cloned_chromosomes = elitist_selection(population, fitnesses, num_to_clone)

    else:
        raise ValueError("Invalid selection method!")

    return cloned_chromosomes


def crossover(parent1, parent2, method = "Uniform", k = 0):
    # Create a new chromosome using uniform crossover of 2 parents
    if method == "Uniform":
        #print("Uniform Crossover")
        child1, child2 = uniform_crossover(parent1, parent2)
        return child1, child2
    elif method == "K-point" and 1 <= k <= 4:
        #print("K-point Crossover")
        child1, child2 = k_point_crossover(parent1, parent2, k)
        return child1, child2
    else:
        raise ValueError("Invalid crossover method!")


def generateRandomChromosomes(population_size):
    generatedChromosomes = []
    for _ in range(population_size):
        generatedChromosomes.append(Chromosome())
    return generatedChromosomes


def parse_args():
    if len(sys.argv) < 9:
        print("Usage: Refer to README.md")
        print("Example:  python .\main.py .\genAlgData1.txt 150 10 Elitist 90 K-point 1 0.50")
        print("For Uniform Search, include any number for k-point number (Does not use k-point)")
        exit(1)

    filename = sys.argv[1]
    population_size = int(sys.argv[2])
    generation_count = int(sys.argv[3])
    selection = sys.argv[4]
    x_percentage = int(sys.argv[5])
    crossover_method = sys.argv[6]
    k_point = int(sys.argv[7])
    mutation_rate = float(sys.argv[8])

    return filename, population_size, generation_count, selection, x_percentage, crossover_method, k_point, mutation_rate

def generation(training_data, population_size, selection, x_percentage, crossover_method, k_point, mutation_rate, chromosomes):
    # Fitness of the population
    fitnesses = [chromosome.compute_fitness(training_data) for chromosome in chromosomes]
    # FIRST PORTION
    # Elitist / Roulette Wheel Selection
    selection_number = int(population_size * x_percentage / 100)
    # Create next generation
    next_generation = create_next_generation(chromosomes, fitnesses, x_percentage, selection, selection_number)
    # print("Next Generation")

    # SECOND PORTION
    selection_number = population_size - selection_number
    # Create next generation
    next_generation_2 = create_next_generation(chromosomes, fitnesses, selection_number, selection, selection_number)
    # print(len(next_generation))
    # print(len(next_generation_2))
    # print("Next Generation")

    # Perform crossover on generation 2
    crossover_children = []
    for i in range(0, len(next_generation_2), 2):
        # crossover_children.append(crossover(next_generation_2[i], next_generation_2[i+1], crossover_method, k_point))
        child1, child2 = crossover(next_generation_2[i], next_generation_2[i + 1], crossover_method, k_point)

        crossover_children.append(child1)
        crossover_children.append(child2)
    # print("Selection children")
    # print(len(next_generation))
    # print("Crossed over children")
    # print(len(crossover_children))

    # Combine both generations
    next_generation.extend(crossover_children)
    # print("Combined children")
    # print(len(next_generation))

    # Mutate the Generation
    mutated_children = mutate_population(next_generation, mutation_rate)
    # print("Mutated children")
    # print(len(mutated_children))

    # Fitness of the mutated children
    mutated_children_fitnesses = [chromosome.compute_fitness(training_data) for chromosome in mutated_children]
    #print("Generation: ", g)
    #print(f"Fitnesses of the mutated children " + str(mutated_children_fitnesses))

    # Print Max, Min, and Average Fitnesses
    # print("Max Fitness: ", max(mutated_children_fitnesses))
    # print("Min Fitness: ", min(mutated_children_fitnesses))
    # print("Average Fitness: ", sum(mutated_children_fitnesses) / len(mutated_children_fitnesses))
    # print("\n")
    return mutated_children



def main():
    filename, population_size, generation_count, selection, x_percentage, crossover_method, k_point, mutation_rate = parse_args()
    training_data = load_training_data(filename)
    best_chromosomes = []
    best_chromosome = Chromosome(0, 0, 0, 0, 0)
    #print(training_data)

    # Initial Population
    chromosomes = generateRandomChromosomes(population_size)
    # for chromosome in chromosomes:
    #     print(vars(chromosome))

    # Compute fitnesses
    #fitnesses = [chromosome.compute_fitness(training_data) for chromosome in chromosomes]
    #print(f"Fitnesses of the population " + str(fitnesses))

    for g in range(generation_count):
        print("Generation: ", g)
        mutated_children = generation(training_data, population_size, selection, x_percentage, crossover_method, k_point, mutation_rate, chromosomes)

        # Fitness of the mutated children
        mutated_children_fitnesses = [chromosome.compute_fitness(training_data) for chromosome in mutated_children]
        # Print Max, Min, and Average Fitnesses
        print("Max Fitness: ", max(mutated_children_fitnesses))
        print("Min Fitness: ", min(mutated_children_fitnesses))
        print("Average Fitness: ", sum(mutated_children_fitnesses) / len(mutated_children_fitnesses))
        print("\n")

        # Store the best chromosomes
        best_chromosomes.append(max(mutated_children_fitnesses))

        # Highest Fitness Chromosome from the Final Generation
        if g == generation_count - 1:
            print("Best Chromosomes From Each Generation:")
            print(best_chromosomes)

            print("Final Generation Best Chromosome: ")
            best_chromosome = mutated_children[mutated_children_fitnesses.index(max(mutated_children_fitnesses))]
            print("Best Chromosome Fitness: ", best_chromosome.compute_fitness(training_data))
            print(vars(best_chromosome))


    #print("Best Chromosomes:")
    #print(best_chromosomes)

        # Store the best chromosome
        #best_chromosome_of_generation = mutated_children[mutated_children_fitnesses.index(max(mutated_children_fitnesses))]
        # Double Check Chromosome Fitness
        #print("Best Chromosome Fitness: ", best_chromosome_of_generation.compute_fitness(training_data))

        #print("Best Chromosome of Generation: ", best_chromosome_of_generation)

        #best_chromosome_of_generation = max(mutated_children_fitnesses)
        #print(vars(best_chromosome_of_generation))
        #best_chromosomes.append(best_chromosome_of_generation)
        #best_chromosomes.append(max(mutated_children_fitnesses))

    # Print the best chromosome
    # print("Best Chromosomes:")
    # print(best_chromosomes)
    # print("Best Chromosome Overall: ")

if __name__ == "__main__":
    main()



