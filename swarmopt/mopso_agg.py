import numpy as np
import copy

class Particle:

    def __init__(self, lb, ub):
        """Initialize the particle.

        Attributes
        ----------
        lb : float
            lower bounds for initial values
        ub : float
            upper bounds for initial values
        """

        self.lb = lb
        self.ub = ub

        self.position = np.random.uniform(lb, ub, size=lb.shape[0])
        self.velocity = np.random.uniform(lb, ub, size=lb.shape[0])
        self.fitness = None

        self.pbest_position = self.position
        self.pbest_fitness = float('inf')


    def move(self):
        self.position += self.velocity


class Swarm:

    def __init__(self, function_list, n_particles, n_iterations,
                 lb, ub, w=0.7, c1=2.0, c2=2.0):
        """Initialize the swarm.

        Attributes
        ---------
        function_list : list
            list of functions to optimize
        n_particles : int
            number of particles in swarm
        n_iterations : int
            number of optimization iterations
        lb : float
            lower bounds for initial values
        ub : float
            upper bounds for initial values
        w : float
            inertia weight
        c1 : float
            cognitive weight
        c2 : float
            social weight
        """

        self.function_list = function_list
        self.n_obj = len(function_list)

        self.n_particles = n_particles
        self.n_iterations = n_iterations

        assert len(lb) == len(ub)
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.gbest_position = np.random.uniform(lb, ub, size=self.lb.shape[0])
        self.gbest_fitness = float('inf')

        self.population = []
        self.iteration = 0


    def reset_environment(self):
        self.population = []
        self.iteration = 0

    def termination_check(self):
        if self.iteration > self.n_iterations:
            return False
        else:
            return True

    def initialise_swarm(self):
        for _ in range(self.n_particles):
            self.population.append(Particle(self.lb, self.ub))

    def eval_fitness(self, particle):
        """Evaluate particle fitness based on all functions in function_list"""

        _fitness = 0

        for func in self.function_list:
            _fitness += func(particle.position)

        particle.fitness = _fitness

    def swarm_eval_fitness(self):
        for particle in self.population:
            self.eval_fitness(particle)

    def update_velocity(self, particle):
        inertia = self.w * particle.velocity
        cognitive = (self.c1 * np.random.uniform()
                     * (particle.pbest_position - particle.position))
        social = (self.c2 * np.random.uniform()
                  * (self.gbest_position - particle.position))

        particle.velocity = inertia + cognitive + social

    def swarm_update_velocity(self):
        for particle in self.population:
            self.update_velocity(particle)

    def update_pbest(self, particle):
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position

    def update_gbest(self, particle):
        if particle.fitness < self.gbest_fitness:
            self.gbest_fitness = copy.deepcopy(particle.fitness)
            self.gbest_position = copy.deepcopy(particle.position)

    def swarm_update_best(self):
        for particle in self.population:
            self.update_pbest(particle)
            self.update_gbest(particle)

    def swarm_move(self):
        for particle in self.population:
            particle.move()

    def optimise(self):

        self.reset_environment()
        self.initialise_swarm()

        while self.termination_check():
            self.swarm_eval_fitness()
            self.swarm_update_best()

            self.swarm_update_velocity()
            self.swarm_move()

            self.iteration += 1

if __name__ == '__main__':

    print('MOPSO: Aggregating Approach')

    def function_one(position):
        return np.square(position[0])

    def function_two(position):
        return np.square(position[0] - 2)

    function_list = [function_one, function_two]

    n_particles = 30
    n_iterations = 100

    lb = [-100]
    ub = [100]

    swarm = Swarm(function_list=function_list,
                  n_particles=n_particles,
                  n_iterations=n_iterations,
                  lb=lb,
                  ub=ub)

    swarm.optimise()

    print('gbest_position: ', swarm.gbest_position)
    print('gbest_fitness:  ', swarm.gbest_fitness)
