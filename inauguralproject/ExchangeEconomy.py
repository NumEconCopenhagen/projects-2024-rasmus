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
            
    def optimize_allocation_pareto_improvement(self):
        """Optimize allocation to maximize A's utility with Pareto improvement constraints."""
        def objective(x):
            # Negative utility for A because we minimize in scipy.optimize
            return -self.utility_A(x[0], x[1])
    
        # Constraints to ensure both A and B are at least as well off as their initial endowments
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.utility_A(x[0], x[1]) - self.utility_A(self.par.w1A, self.par.w2A)},
            {'type': 'ineq', 'fun': lambda x: self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)}
        ]
    
        # Bounds to ensure allocations are within feasible range
        bounds = ((0, 1), (0, 1))
    
        # Initial guess (starting point of the optimization algorithm)
        x0 = [self.par.w1A, self.par.w2A]
    
        # Perform the optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
        if result.success:
            return result.x, -result.fun
        else:
            raise ValueError("Optimization failed.")
    def maximize_utility_unrestricted(self):
        """Utility maximization for Consumer A without restrictions other than B's initial utility."""
        # Objective function: Negative of A's utility to turn the maximization problem into a minimization problem
        def objective(x):
            return -self.utility_A(x[0], x[1])

        # Constraint for B's utility to be at least as high as the initial utility
        def constraint(x):
            return self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)

        # Initial guess for the allocation
        x0 = [self.par.w1A, self.par.w2A]

        # Bounds for the allocations
        bounds = ((0, 1), (0, 1))

        # Constraint dictionary
        cons = {'type': 'ineq', 'fun': constraint}

        # Solve the optimization problem
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        if result.success:
            optimal_allocation = result.x
            print(f"The optimal allocation for A is: ({optimal_allocation[0]:.4f}, {optimal_allocation[1]:.4f})")
            print(f"The optimal allocation for B is: ({1-optimal_allocation[0]:.4f}, {1-optimal_allocation[1]:.4f})")
            print(f"A's utility: {self.utility_A(optimal_allocation[0], optimal_allocation[1]):.4f}")
            print(f"B's utility: {self.utility_B(1-optimal_allocation[0], 1-optimal_allocation[1]):.4f}")
            print(f"Total utility: {self.utility_A(optimal_allocation[0], optimal_allocation[1]) + self.utility_B(1-optimal_allocation[0], 1-optimal_allocation[1]):.4f}")
        else:
            print("Optimization was not successful.")
    def maximize_aggregate_utility(self):
        """Maximize the aggregate utility of consumers A and B."""
        # Define the objective function for aggregate utility
        def objective(x):
            # Calculate B's consumption based on A's consumption
            x1B, x2B = 1 - x[0], 1 - x[1]
            # Aggregate utility is the sum of A's and B's utilities
            return -(self.utility_A(x[0], x[1]) + self.utility_B(x1B, x2B))

        # Initial guess for A's allocation could be their initial endowments
        x0 = [self.par.w1A, self.par.w2A]

        # Bounds to ensure allocations are within the feasible range
        bounds = ((0, 1), (0, 1))

        # Perform the optimization to maximize aggregate utility
        result = minimize(objective, x0, method='SLSQP', bounds=bounds)

        if result.success:
            optimal_allocation_A = result.x
            optimal_allocation_B = 1 - result.x
            return optimal_allocation_A, optimal_allocation_B, self.utility_A(*optimal_allocation_A) + self.utility_B(*optimal_allocation_B)
        else:
            raise ValueError("Optimization failed to maximize aggregate utility.")
    def maximize_total_utility(self):
        """Maximize the total utility of consumers A and B."""
        # Define the objective function for total utility
        def objective(x):
            # Calculate B's consumption based on A's consumption
            x1B, x2B = 1 - x[0], 1 - x[1]
            # Total utility is the sum of A's and B's utilities
            return -(self.utility_A(x[0], x[1]) + self.utility_B(x1B, x2B))

        # Initial guess for A's allocation could be their initial endowments
        x0 = [self.par.w1A, self.par.w2A]

        # Bounds to ensure allocations are within the feasible range
        bounds = ((0, 1), (0, 1))

        # Perform the optimization to maximize total utility
        result = minimize(objective, x0, method='SLSQP', bounds=bounds)

        if result.success:
            optimal_allocation_A = result.x
            optimal_allocation_B = 1 - result.x
            return optimal_allocation_A, optimal_allocation_B, self.utility_A(*optimal_allocation_A) + self.utility_B(*optimal_allocation_B)
        else:
            raise ValueError("Optimization failed to maximize total utility.")