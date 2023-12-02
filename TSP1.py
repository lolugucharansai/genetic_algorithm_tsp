import numpy as np
import random

def calculate_total_distance(tour, graph):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += graph[tour[i]][tour[i + 1]]
    total_distance += graph[tour[-1]][tour[0]]
    return total_distance
#partially mapped crossover
def pmx_crossover(parent1, parent2):
    # Get the size of the parents
    size = len(parent1)

    # Initialize mapping lists for both parents
    p1, p2 = [0]*size, [0]*size

    # Fill the mapping lists with the indices of each city in the parents
    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i

    # Generate two random crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: 
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Swap the sections between the crossover points of the two parents
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the cities to be swapped
        temp1 = parent1[i]
        temp2 = parent2[i]

        # Swap the cities in the parents
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2

        # Update the mapping lists
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    # Return the offspring
    return parent1, parent2

def mutate(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]

def create_initial_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

def calculate_fitness(population, graph):
    return [1 / calculate_total_distance(tour, graph) for tour in population]

# selects the num number of elements from the population with the probability proportional to the fitness scores
def select(population, fitness, num):
    return random.choices(population, weights=fitness, k=num)

def generate_next_generation(population, graph, elite_size, mutation_rate):
    fitness = calculate_fitness(population, graph)
    elite = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:elite_size]
    elite = [x[0] for x in elite]
    selected = select(population, fitness, len(population) - elite_size)
    children = []
    for i in range(0, len(selected), 2):
        child1, child2 = pmx_crossover(selected[i], selected[i+1])
        children.append(child1)
        children.append(child2)
    for child in children:
        if random.random() < mutation_rate:
            mutate(child)
    
    return elite + children
    

def genetic_algorithm(graph, population_size, elite_size, mutation_rate, generations):
    population = create_initial_population(population_size, len(graph))
    for _ in range(generations):
        population = generate_next_generation(population, graph, elite_size, mutation_rate)
        fitness = calculate_fitness(population, graph)
        avg_fitness = sum(fitness) / population_size
        print(f"Average fitness in generation {_ + 1}: {avg_fitness}")
    best_tour = min(population, key=lambda x: calculate_total_distance(x, graph))
    best_distance = calculate_total_distance(best_tour, graph)
    return best_tour, best_distance

# Test the function
graph = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]#A complete Graph (if incomplte graph modify the mutation funtion accordingly).
best_tour, best_distance = genetic_algorithm(graph, population_size=100, elite_size=20, mutation_rate=0.01, generations=50)
print(f"Best tour: {best_tour}")
print(f"Best distance: {best_distance}")
