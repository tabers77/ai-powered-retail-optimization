import numpy as np
import random
from typing_extensions import TypedDict
from typing import NamedTuple, List
from collections import defaultdict
import scipy.stats as stats
import matplotlib.pyplot as plt


def select_alpha_beta_priors(prices: np.array, demands: np.array, method: str = 'uniform'):
    alpha, beta = None, None

    if method == 'uniform':
        alpha = np.ones_like(prices)  # uniform distribution with alpha=1
        beta = np.ones_like(prices)  # uniform distribution with beta=1

    elif method == 'mean_based':
        alpha, beta = demands.mean(), 1  # demands.std()

    elif method == 'informative_priors':
        alpha = np.ones_like(prices) + demands
        beta = np.ones_like(prices) + np.sum(demands) - demands

    elif method == 'exponential':
        alpha = np.exp(np.mean(np.log(demands))) * np.ones_like(
            prices)  # exponential distribution with lambda=mean(demands)
        beta = np.ones_like(prices)  # uniform distribution with beta=1

    return alpha, beta


def beta_dist(alpha, beta):
    return np.random.beta(alpha, beta)


class PriceParams(TypedDict):
    price: float
    alpha: float
    beta: float


def get_p_lambdas(prices_to_test, alpha_0, beta_0):
    p_lambdas = []
    for price in prices_to_test:
        p_lambdas.append(
            PriceParams(
                price=price,
                alpha=alpha_0,
                beta=beta_0
            )
        )
    return p_lambdas


class OptimalPriceResult(NamedTuple):
    price: float
    price_index: int


class ThomsonSampler:
    """
     Class implementing the Thomson Sampling algorithm for price optimization.

     Parameters:
         prices_to_test (List[float]): List of prices to test.
         demands (List[float]): List of demand values corresponding to the prices.

     Methods:
         baseline1(n_iterations=100, prior_init_type='informative_priors'):
             Executes the Thomson Sampling algorithm with flat or informative priors.
             Returns the optimal price based on the final demand probabilities.

         baseline2(n_iterations=100, prior_init_type='mean_based', visualize_samples=True):
             Executes the Thomson Sampling algorithm with mean-based priors and gamma distribution.
             Returns a dictionary mapping prices to their respective revenue probabilities.
     """
    def __init__(self, prices_to_test, demands):
        """
        Initialize the ThomsonSampler instance.

        Args:
            prices_to_test (List[float]): List of prices to test.
            demands (List[float]): List of demand values corresponding to the prices.
        """
        self.prices_to_test = np.array(prices_to_test)
        self.demands = np.array(demands)

    def baseline1(self, n_iterations=100, prior_init_type='informative_priors'):
        """
        Executes the Thomson Sampling algorithm with flat or informative priors.

        Args:
            n_iterations (int): Number of iterations to run the algorithm (default: 100).
            prior_init_type (str): Type of priors initialization: 'uniform', 'mean_based', or 'informative_priors'
                (default: 'informative_priors').

        Returns:
            float: The optimal price based on the final demand probabilities.
        """
        # ----------------------
        # 1. Define the priors
        # ----------------------
        # parameters for the beta distribution - NOT FLAT PRIORS
        alpha_prior, beta_prior = select_alpha_beta_priors(prices=self.prices_to_test, demands=self.demands,
                                                           method=prior_init_type)

        # Run the Thomson Sampling algorithm for 1000 iterations
        choices = list()

        for i in range(n_iterations):
            # Here we define our priors
            # Sample demand probabilities for each price point
            print('-------------------')
            print(f'NEW ITERATION: {i + 1}')
            print('-------------------')
            # ---------------------------------------------
            # 2. Compute beta distributions based on priors
            # These parameters are updated based on exploration and exploitation
            # ---------------------------------------------
            demand_probs = beta_dist(alpha_prior, beta_prior)

            print('alpha_prior', list(alpha_prior))
            print('beta_prior', list(beta_prior))
            # ---------------------------
            # 3. Choose the price with the highest demand probability
            # ---------------------------
            price_index = np.argmax(demand_probs)
            price = self.prices_to_test[price_index]
            choices.append(price)

            print('1. Demand_probs', demand_probs)
            print('2. Optima price', price)
            # ---------------------------------
            # 4. Observe the actual demand and update the corresponding beta distribution
            # ---------------------------------
            print(f'2.1 N products sold to determine demand: {self.demands[price_index]}')
            # demand = np.random.binomial(n=demands[price_index], p=0.5)
            demand = random.choice([0, 1])

            # -------------------------------
            # 5. Exploration vs Exploitation
            # -------------------------------
            if demand:
                print('3.demand', demand, '--> EXPLOIT')
                alpha_prior[price_index] += 1
            else:
                print('3.demand', demand, '--> EXPLORE')
                beta_prior[price_index] += 1

            print('4.alpha_prior', list(alpha_prior))
            print('5.beta_prior', list(beta_prior))

        # Recommend the optimal price based on the final beta distributions

        demand_probs = beta_dist(alpha_prior, beta_prior)

        print('final_alpha_prior', list(alpha_prior))
        print('final_beta_prior', list(beta_prior))
        print('final_demand_probs', demand_probs)
        optimal_price = self.prices_to_test[np.argmax(demand_probs)]
        print("Optimal price:", optimal_price)

        return optimal_price

    def baseline2(self, n_iterations=100, prior_init_type='mean_based', visualize_samples=True):
        """
        Executes the Thomson Sampling algorithm with mean-based priors and gamma distribution.

        Args:
            n_iterations (int): Number of iterations to run the algorithm (default: 100).
            prior_init_type (str): Type of priors initialization: 'uniform', 'mean_based', or 'informative_priors'
                (default: 'mean_based').
            visualize_samples (bool): Whether to visualize the distribution samples during iterations (default: True).

        Returns:
            dict: A dictionary mapping prices to their respective revenue probabilities.
        """
        # ----------------------
        # 1. Define the priors
        # ----------------------

        alpha_0, beta_0 = select_alpha_beta_priors(prices=self.prices_to_test, demands=self.demands,
                                                   method=prior_init_type)

        p_lambdas = get_p_lambdas(self.prices_to_test, alpha_0, beta_0)

        def get_optimal_price(prices: List[float], demands: List[float]) -> OptimalPriceResult:
            print(f'get_optimal_price values from prices: {prices} and demands {demands}')
            revenue_ = prices * demands
            print('Result:', revenue_)
            index = np.argmax(revenue_)
            return OptimalPriceResult(price_index=index, price=prices[index]), revenue_

        def sample_demands_from_model(p_lambdas: List[PriceParams]) -> List[float]:
            return list(map(lambda v: np.random.gamma(v['alpha'], 1 / v['beta']), p_lambdas))

        def plot_distributions(p_lambdas: List[PriceParams], iteration: int):
            # TODO: ADAPT THIS PART DYNAMICALLY , AND MAKE IT OPTIONAL
            x = np.arange(0, np.array(self.demands).max() + 5, 0.10)
            for dist in p_lambdas:
                y = stats.gamma.pdf(x, a=dist["alpha"], scale=1 / dist["beta"])
                plt.plot(x, y, label=dist["price"])
                plt.xlabel("demand")
                plt.ylabel("pdf")
            plt.title(f"PDFs after Iteration: {iteration}")
            plt.legend(loc="upper right")
            plt.show()

        # Thompson sampling for solving the explore-exploit dilemma.

        price_counts = defaultdict(lambda: 0)
        revenues = list()
        for t in range(n_iterations):
            print('---------------')
            print('Iteration', t + 1)
            print('---------------')
            # -------------------------------
            # 2. Exploration vs Exploitation
            # Defining demands from alpha and beta
            # -------------------------------
            if random.choice([1, 0]) == 0:
                print('2. EXPLOITATION')
                demands = self.demands.copy()
            else:
                print('2. EXPLORATION')
                demands = sample_demands_from_model(p_lambdas)
            print('demands:', demands)
            # -------------------------------
            # 3. Get optimal price
            # Using the current demands multiply demand * price
            # -------------------------------
            optimal_price_res, revenue = get_optimal_price(self.prices_to_test, demands)
            revenues.append(revenue)
            print('optimal_price_res:', optimal_price_res)

            # -------------------------------
            # 4. increase the count for the price
            # -------------------------------
            price_counts[optimal_price_res.price] += 1
            print('price_counts status:', price_counts)

            # -------------------------------
            # 5. Define parameter v
            # v is used to generate demands when exploring
            # -------------------------------
            # update model parameters
            v = p_lambdas[optimal_price_res.price_index]
            v['alpha'] += demands[optimal_price_res.price_index]
            v['beta'] += 1
            if visualize_samples:
                if t % 10 == 0:
                    plot_distributions(p_lambdas, t)

        revenues_probs = [i / sum(revenues[-1]) for i in revenues[-1]]

        return dict(zip(self.prices_to_test, revenues_probs))