# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from universal import tools
from universal.algo import Algo


class OLU(Algo):
    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, eta=26, beta=0.1, gama=0.001, **kwargs):
        """
        :param window: Lookback window.
        :param eps:
        :param theta:
        """

        super().__init__(min_history=1, **kwargs)

        self.eta = eta
        self.gama = gama
        self.beta = beta
        self.phi = np.array([])


    def init_step(self, X):
        # set initial phi to x1
        self.phi = X.iloc[1, :] / X.iloc[0, :]

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def predict(self, hist):
        """Predict next price relative."""
        return hist.mean() / hist.iloc[-1, :]


    def calculate_formula(self, t, l, tau, history):
        result = 0
        div=0
        for i in range(t - l + 1, t + 1):
            result += np.exp(-(t + 1 - i) ** 2 * history.iloc[i,:] / (2 * tau ** 2)) / (2 * tau ** 2 )
            div += np.exp(-(t + 1 - i) ** 2 / (2 * tau ** 2)) / (2 * tau ** 2 )
        return result/div

    def step(self, x, last_b, history):

        self.phi = history.iloc[-1, :] / history.iloc[-2, :]

        #self.phi=self.keep_top_five(self.phi)

        # if self.day>5:
        #     x_pre=self.calculate_formula(t=self.day-1, l=5, tau=2.8, history=history)
        #     self.phi = x_pre / history.iloc[-1, :]
        # else:
        #     self.phi=self.predict(history)

        alpha = 30
        # Update the weights

        b = self.OLU(pt=last_b, xt=self.phi, eta=self.eta, alpha=alpha, beta=self.beta)


        return b

    # def predict(self, hist):
    #     """Predict next price relative."""
    #     return hist.mean() / hist.iloc[-1, :]

    def OLU(self, pt, xt, eta, alpha, beta, max_iter=10000, tol=1e-4):
        # Initialize variables
        n = len(pt)
        p = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)


        # def objective_function(p):
        #     term1 = -eta * (np.log(np.dot(p.T, xt)) + np.dot(xt.T, (p - pt)) / np.dot(p.T, xt))
        #     term2 = 0.5 * np.linalg.norm(p - pt, ord=2) ** 2
        #     term3 = 0.5 * beta * np.linalg.norm(p - pt, ord=2) ** 2
        #     return term1+term2+term3

        # def constraint1(x):
        #     return np.sum(x) - 1
        #
        # def constraint2(x):
        #     return np.dot(x.T,xt)
        #
        # con1 = {'type': 'eq', 'fun': constraint1}
        # con2 = {'type': 'ineq', 'fun': constraint2}
        #
        # con=[]
        # con.append(con1)
        # con.append(con2)
        # cons=tuple(con)

        # ADMM iterations
        for k in range(max_iter):


            z_prev=np.copy(z)


            # # 设置初始权重
            # x0 = len(xt) * [1 / len(xt)]
            #
            # # 规划求解
            # solution = minimize(objective_function, x0, method='SLSQP', constraints=cons)


            # Update p
            # p=solution.x
            # p=tools.simplex_proj(p)
            p = tools.simplex_proj(
                y=- eta * xt / ((beta + 1) * np.dot(pt, xt)) + pt + beta * z / (beta + 1) - beta * u / (beta + 1)
            )


            # Update z
            z = shrinkage_operator(p - pt + u, alpha / beta)


            # Update u
            u = u + p - pt - z

            # Stopping criteria
            primal_residual = np.linalg.norm(p - pt - z)
            dual_residual = np.linalg.norm(alpha * (z - z_prev))

            # Stopping criteria
            if primal_residual < tol and dual_residual < tol:
                #print(f"Converged after {k + 1} iterations.")
                break

        return p



def shrinkage_operator(y, rho):
    # Shrinkage operator
    return np.maximum(0, y - rho) - np.maximum(0, -y - rho)


if __name__ == "__main__":
    tools.quickrun(OLU(eta=26, beta=0.1, gama=0.001))
