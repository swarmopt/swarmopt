import numpy as np
import copy

class Particle:

    def __init__(self, lb, ub, n_obj):

        """Initialise the particle.

        Attributes
        ----------
        lb : array
            lower bounds for initial values
        ub : array
            upper bounds for initial values
        n_obj : int
            number of objectives to evaluate
        """

        self.lb = lb
        self.ub = ub
        self.n_obj = n_obj

        self.position = np.random.uniform(lb, ub, size=lb.shape[0])
        self.velocity = np.random.uniform(lb, ub, size=lb.shape[0])
        self.fitness = [None] * n_obj

        self.pbest_position = self.position
        self.pbest_fitness = [float('inf')] * n_obj

    def dominate(self, opponent):

        """
        Returns True if this particle dominates.
        Returns False if opponent dominates.
        """

        self_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < opponent.fitness[i]:
                self_dominates = True
            elif opponent.fitness[i] < self.fitness[i]:
                return False

        return self_dominates

    def self_dominate(self):

        """
        Returns True if current state dominates.
        Returns False if pbest dominates.
        """

        new_fitness_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < self.pbest_fitness[i]:
                new_fitness_dominates = True
            elif self.pbest_fitness[i] < self.fitness[i]:
                return False

        return new_fitness_dominates

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
        self.gbest_fitness = [float('inf')] * self.n_obj

        self.population = []
        self.archive = []

        self.iteration = 0

    # Base Functions
    def reset_environment(self):
        self.iteration = 0
        self.population = []
        self.archive = []

    def termination_check(self):
        if self.iteration > self.n_iterations:
            return False
        else:
            return True

    def initialise_swarm(self):
        for _ in range(self.n_particles):
            self.population.append(Particle(self.lb, self.ub, self.n_obj))

    # Update Functions
    def eval_fitness(self, particle):
        for idx, function in enumerate(self.function_list):
            particle.fitness[idx] = function(particle.position)

    def swarm_eval_fitness(self):
        for particle in self.population:
            self.eval_fitness(particle)

    def update_pbest(self, particle):
        if particle.self_dominate():
            particle.pbest_position = particle.position
            particle.pbest_fitness = particle.fitness

    def swarm_update_pbest(self):
        for particle in self.population:
            self.update_pbest(particle)

    def update_velocity(self, particle):
        _leader = self.choose_leader()

        inertia = self.w * particle.velocity
        cognitive = (self.c1 * np.random.uniform()
                     * (particle.pbest_position - particle.position))
        social = (self.c2 * np.random.uniform()
                  * (_leader.position - particle.position))

        particle.velocity = inertia + cognitive + social

    def swarm_update_velocity(self):
        for particle in self.population:
            self.update_velocity(particle)

    def choose_leader(self):
        """Basic leader selection method."""
        return copy.deepcopy(np.random.choice(self.archive))

    def swarm_move(self):
        for particle in self.population:
            particle.move()

    # Archive Based Functions
    @staticmethod
    def pareto_front(population):

        """Returns the Pareto Front of the supplied population."""

        _population = copy.deepcopy(population)

        pf = []

        for idx, particle in enumerate(_population):
            pf.append(particle)

            for opp_idx, opp_particle in enumerate(pf[:-1]):
                if opp_particle.dominate(particle):
                    del pf[-1]
                    break
                elif particle.dominate(opp_particle):
                    del pf[opp_idx]

        return pf

    def optimise(self):

        self.reset_environment()
        self.initialise_swarm()

        while self.termination_check():

            self.swarm_eval_fitness()
            self.swarm_update_pbest()

            self.archive.extend(self.population)
            self.archive = self.pareto_front(self.archive)

            self.swarm_update_velocity()
            self.swarm_move()

            print('Iteration: {0:03} | Archive Length: {1:05}'.format(
                self.iteration, len(self.archive))
            )

            self.iteration += 1




if __name__ == '__main__':

    print('MOPSO: Archive Approach')

    def function_one(position):
        return np.square(position[0])

    def function_two(position):
        return np.square(position[0] - 2)

    function_list = [function_one, function_two]

    n_particles = 30
    n_iterations = 50

    lb = [-100]
    ub = [100]

    swarm = Swarm(function_list=function_list,
                  n_particles=n_particles,
                  n_iterations=n_iterations,
                  lb=lb,
                  ub=ub)

    swarm.optimise()

    func1 = []
    func2 = []

    for particle in swarm.archive:
        func1.append(particle.fitness[0])
        func2.append(particle.fitness[1])

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,5))

    plt.scatter(func1, func2, s=2, c='r')

    plt.xlim(0,)
    plt.ylim(0,)

    plt.title('Pareto Front: Schaffer Function')
    plt.xlabel('Function One')
    plt.ylabel('Function Two')

    plt.show()










