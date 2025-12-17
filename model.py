# from numpy.typing import NDArray
import functools
import math
from math import comb
from typing import List, Tuple

import networkx as nx
import numpy as np
import numpy.random as rd

# rd = np.random.default_rng()
from scipy.stats import beta
from tqdm.auto import tqdm

from agents import Bandit, BetaAgent


class Model:
    """
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py
    Represents an agent in a network epistemology playground.

    Attributes:
    - network: The network.
    - n_experiments (int): The number of experiments per step.
    - agent_type (str): The type of agents, "bayes", "beta" or "jeffrey"
    - uncertainty (float): The uncertainty in the experiment.
    - p_theories (list): The success probabilities of the theories.

    Methods:
    - __init__(self): Initializes the Model object.
    - __str__(self): Returns a string representation of the Model object.
    - simulation(self, number_of_steps): Runs a simulation of the model.
    - step(self): Updates the model with one step, consisting of experiments and
    updates.
    - agents_experiment(self): Updates the model with one round of experiments.
    - agents_update(self): Updates the model with one round of updates.
    """

    def __init__(
        self,
        network: nx.DiGraph | nx.Graph,
        n_experiments: int,
        uncertainty: float = 0.001,
        tolerance: float | None = 5 * 1e-03,
        histories=False,
        sampling_update=False,
        variance_stopping=False,
        seed: int | None = np.random.randint(0, 2**31 - 1),
        *args,
        **kwargs
    ):
        self.network = network
        self.n_agents = network.number_of_nodes()
        self.directedness = nx.is_directed(network)
        # print(self.n_agents)
        self.n_experiments = n_experiments
        # else:
        # self.seed = seed
        if seed is not None:
            rd.seed(seed)
        self.bandit = Bandit(uncertainty)
        self.agents = [
            BetaAgent(
                id, self.bandit, histories=histories, sampling_update=sampling_update
            )
            for id in self.network.nodes()
        ]
        # self.init_agents_alphas_betas = "here goes the list of initial alphas and betas"
        # self.init_agents_alphas_betas= [copy.deepcopy(agent.alphas_betas) for agent in self.agents]

        # agent.id is the name of the node in the network
        # Compute degree centrality
        # degree_centrality_dict = nx.degree_centrality(self.network)
        # # Convert to a NumPy array (vector)
        # degree_centrality_vector = np.array(list(degree_centrality_dict.values()))
        # self.degree_centrality_vector = degree_centrality_vector
        # self.degree_centrality_vector = "here goes the degree centrality vector"

        # self.agent_type = agent_type
        self.step_counter = 0
        self.tolerance: float | None = tolerance
        self.histories = histories
        self.variance_stopping = variance_stopping

    def run_simulation(
        self, n_steps: int = 10**4, show_bar: bool = False, *args, **kwargs
    ):
        """Runs a simulation of the model and sets model.conclusion.

        Args:
            number_of_steps (int, optional): Number of steps in the simulation
            (it will end sooner if the stop condition is met). Defaults to 10**4."""

        def stop_condition(credences_prior, credences_post) -> bool:
            # the tolerance is too tight, originally: rtol=1e-05, atol=1e-08
            return False
            return np.allclose(
                credences_prior,
                credences_post,
                rtol=self.tolerance,
                atol=self.tolerance,
            )

        def determine_conclusion() -> float:
            # Count how many pairs have the second coordinate larger than the first
            # (second coordinate is the second theory)
            credences = np.array([agent.credences for agent in self.agents])
            counts = np.sum([pair[1] > pair[0] for pair in credences])
            return counts / len(credences)  # (second_coordinates > 0.5).mean()

        iterable_n_steps = range(n_steps)

        if show_bar:
            iterable_n_steps = tqdm(iterable_n_steps)

        for _ in iterable_n_steps:
            # Lots of if elses but oh well
            if self.variance_stopping:
                betas_prior = np.array([agent.alphas_betas for agent in self.agents])
                # mv_prior = np.array([beta.stats(prior[0], prior[1], moments='mv') for
                # prior in betas_prior])
                mv_prior = np.array(
                    [
                        [
                            beta.stats(prior[0][0], prior[0][1], moments="mv"),
                            beta.stats(prior[1][0], prior[1][1], moments="mv"),
                        ]
                        for prior in betas_prior
                    ]
                )
            else:
                credences_prior = np.array([agent.credences for agent in self.agents])

            self.step()

            if self.variance_stopping:
                betas_post = np.array([agent.alphas_betas for agent in self.agents])
                # mv_post = np.array([beta.stats(post[0], post[1], moments='mv') for post in betas_post])
                mv_post = np.array(
                    [
                        [
                            beta.stats(post[0][0], post[0][1], moments="mv"),
                            beta.stats(post[1][0], post[1][1], moments="mv"),
                        ]
                        for post in betas_post
                    ]
                )

            # we need the credences post regardless of the variance stopping condition
            credences_post = np.array([agent.credences for agent in self.agents])

            # if self.step_counter > 1000 and self.step_counter % 100 == 0:
            #     if self.prob_some_agent_switches() < 0.01:
            #         break

            # if self.variance_stopping:
            #     if stop_condition(mv_prior, mv_post):
            #         break
            # else:
            #     if stop_condition(credences_prior, credences_post):
            #         break

        # We add the conclusion at the end of the simulation
        self.conclusion = determine_conclusion()

        if self.histories:
            self.add_agents_history()

    def step(self):
        """Updates the model with one step, consisting of experiments and updates."""
        self.step_counter += 1
        experiments_results = self.agents_experiment()
        self.agents_update(experiments_results)

    def agents_experiment(self) -> dict[object, list[int]]:
        experiments_results = {}
        for agent in self.agents:
            theory_index, n_success, n_failures = agent.experiment(self.n_experiments)
            experiments_results[agent.id] = [theory_index, n_success, n_failures]
        # print('experiments done')
        return experiments_results

    def agents_update(self, experiments_results: dict[object, list[int]]) -> None:
        for agent in self.agents:
            # Gather information from predecessors
            if nx.is_directed(self.network):
                neighbor_nodes = list(self.network.predecessors(agent.id))  # type: ignore
            else:
                neighbor_nodes = list(self.network.neighbors(agent.id))
            theories_exp_results = np.array([np.array([0, 0]), np.array([0, 0])])
            # for reference: experiments_results[agent.id]=[theory_index, n_success, n_failures]
            results = experiments_results[agent.id]
            theory_index = results[0]
            theories_exp_results[theory_index][0] += results[1]
            theories_exp_results[theory_index][1] += results[2]
            for id in neighbor_nodes:
                results = experiments_results[id]
                theory_index = results[0]
                theories_exp_results[theory_index][0] += results[1]  # n_success
                theories_exp_results[theory_index][1] += results[2]  # n_failures

            # update
            agent.beta_update(0, theories_exp_results[0][0], theories_exp_results[0][1])
            agent.beta_update(1, theories_exp_results[1][0], theories_exp_results[1][1])

    def add_agents_history(self):
        self.agent_histories = [agent.credences_history for agent in self.agents]

    ############################################
    # Code below is not used
    ############################################
    def prob_switch_exact(
        self, agent: BetaAgent, batch_size=1, tol=1e-12, max_rounds=2000
    ):
        """
        Compute the probability that an agent ever switches arms.

        Parameters
        ----------
        betas : tuple ((a0,b0), (a1,b1))
            Beta parameters for the two arms.
        true_rates : tuple (s0, s1)
            True success rates for arms 0 and 1.
        n : int
            Batch size.
        tol : float
            Tolerance for convergence.
        max_rounds : int
            Safety bound.

        Returns
        -------
        float
            Probability of EVER switching arms.
        """

        # Unpack betas and true success rates
        (a0, b0), (a1, b1) = agent.alphas_betas
        s0, s1 = agent.bandit.p_bad_theory, agent.bandit.p_good_theory

        # Posterior means
        mu0 = a0 / (a0 + b0)
        mu1 = a1 / (a1 + b1)

        # Determine currently sampled arm
        if mu1 > mu0:
            # Sample arm 1
            a_s, b_s = a1, b1
            mu_o = mu0
            p_s = s1
        else:
            # Sample arm 0
            a_s, b_s = a0, b0
            mu_o = mu1
            p_s = s0

        # =======================================================================
        #   DP recursion for switching probability
        # =======================================================================

        @functools.lru_cache(None)
        def dp(a, b):
            """
            DP(a,b) = switching probability starting from Beta(a,b)
            for the sampled arm, with other-arm mean mu_o fixed.
            """
            # If true rate is already <= competitor mean: certain switch
            if p_s <= mu_o:
                return 1.0

            # If posterior mean drops below other-arm mean: switch now
            if a / (a + b) <= mu_o:
                return 1.0

            # Otherwise compute expected value over batch outcomes
            pswitch = 0.0

            for k in range(batch_size + 1):
                # P(s=k successes)
                pk = comb(batch_size, k) * (p_s**k) * ((1 - p_s) ** (batch_size - k))
                a2 = a + k
                b2 = b + (batch_size - k)

                # Check if switch occurs after this batch
                if a2 / (a2 + b2) <= mu_o:
                    pswitch += pk
                else:
                    pswitch += pk * dp(a2, b2)

            return pswitch

        # We simply return dp evaluated at the initial sampled-arm parameters
        p0 = dp(a_s, b_s)
        return p0

    def prob_some_agent_switches(self, batch_size: int = 1, tol=1e-12) -> float:
        """
        Parameters
        ----------
        agents : list of tuples (a0, b0, a1, b1)
            Beta priors for each agent.
        S0, S1 : floats
            True success rates of arm 0 and 1 respectively.
        n : int
            Batch size.
        tol : float
            DP tolerance.

        Returns
        -------
        individual_probs : list of floats
            Switching probability for each agent.
        p_at_least_one : float
            Probability at least one agent ever switches.
        """

        individual = []
        # agents_betas = [agent.alphas_betas for agent in self.agents]

        for agent in self.agents:
            #     [a0, b0], [a1, b1] = agent.alphas_betas
            # # for [[a0, b0], [a1, b1]] in agents_betas:
            #     # Posterior means
            #     mu0 = a0 / (a0 + b0)
            #     mu1 = a1 / (a1 + b1)

            #     # Determine which arm agent samples initially
            #     if mu1 > mu0:
            #         # Sample arm 1
            #         a_s, b_s = a1, b1
            #         mu_o = mu0
            #         p_s = self.bandit.p_good_theory
            #     else:
            #         # Sample arm 0
            #         a_s, b_s = a0, b0
            #         mu_o = mu1
            #         p_s = self.bandit.p_bad_theory

            # Exact switching probability
            try:
                p_switch = self.prob_switch_exact(agent, batch_size=batch_size, tol=tol)
            except RecursionError:
                p_switch = 1.0
            # return 1.0
            # p_switch = self.prob_switch_exact(agent, batch_size=batch_size, tol=tol)
            individual.append(p_switch)

        # Probability at least one switches (independent processes)
        p_none = math.prod(1 - p for p in individual)
        p_any = 1 - p_none

        return p_any
