from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar, minimize

class ExchangeEconomyClass:
    def __init__(self, w1A=0.8, w2A=0.3):
        self.par = SimpleNamespace(alpha=1/3, beta=2/3, w1A=w1A, w2A=w2A, p2=1)

    def utility_A(self, x1A, x2A):
        """Calculate utility for consumer A."""
        return (x1A ** self.par.alpha) * (x2A ** (1 - self.par.alpha))

    def utility_B(self, x1B, x2B):
        """Calculate utility for consumer B."""
        return (x1B ** self.par.beta) * (x2B ** (1 - self.par.beta))

    def demand_A(self, p1):
        """Calculate demand for consumer A given price p1."""
        income_A = self.par.w1A * p1 + self.par.w2A * self.par.p2
        x1A_star = self.par.alpha * (income_A / p1)
        x2A_star = (1 - self.par.alpha) * (income_A / self.par.p2)
        return x1A_star, x2A_star

    def demand_B(self, p1):
        """Calculate demand for consumer B given price p1."""
        income_B = (1 - self.par.w1A) * p1 + (1 - self.par.w2A) * self.par.p2
        x1B_star = self.par.beta * (income_B / p1)
        x2B_star = (1 - self.par.beta) * (income_B / self.par.p2)
        return x1B_star, x2B_star

    def market_clearing_error(self, p1):
        """Calculate market clearing error given price p1."""
        x1A_star, x2A_star = self.demand_A(p1)
        x1B_star, x2B_star = self.demand_B(p1)
        eps1 = abs(x1A_star + x1B_star - 1)  # Error for good 1
        eps2 = abs(x2A_star + x2B_star - 1)  # Error for good 2
        return eps1 + eps2  # Total market clearing error

    def find_market_clearing_price(self):
        """Find the market-clearing price p1."""
        result = minimize_scalar(self.market_clearing_error, bounds=(0.01, 10), method='bounded')
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed to find a market-clearing price.")

    def maximize_consumer_A_utility_discrete(self, P1):
        """Maximize consumer A's utility over a discrete set of prices P1."""
        max_utility = float('-inf')
        optimal_price = None
        optimal_allocation_A = None
        for p1 in P1:
            x1B_star, x2B_star = self.demand_B(p1)
            x1A_star, x2A_star = 1 - x1B_star, 1 - x2B_star
            utility_A = self.utility_A(x1A_star, x2A_star)
            if utility_A > max_utility:
                max_utility = utility_A
                optimal_price = p1
                optimal_allocation_A = (x1A_star, x2A_star)
        return optimal_price, optimal_allocation_A, max_utility

    def maximize_consumer_A_utility_continuous(self):
        """Maximize consumer A's utility over a continuous range of prices."""
        result = minimize(lambda p1: -self.utility_A(*(1 - np.array(self.demand_B(p1[0])))), x0=[1], bounds=[(0.01, None)], method='L-BFGS-B')
        if result.success:
            optimal_price = result.x[0]
            optimal_allocation_A = (1 - np.array(self.demand_B(optimal_price)))
            return optimal_price, optimal_allocation_A, -result.fun
        else:
            raise ValueError("Optimization failed to maximize consumer A's utility.")
