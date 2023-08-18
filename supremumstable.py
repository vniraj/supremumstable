
from scipy.special import lambertw
import numpy as np
import positivestable as ps

class SupStable:
    def __init__(self, alpha, beta, generator = None):
        self.alpha = alpha
        self.beta = beta
        assert 0 < self.alpha <= 2 and -1 <= self.beta <= 1 and not (0 < alpha <= 1 and beta == -1)
        self.Delta_0 = 0
        self.gen = np.random.default_rng() if generator == None else generator
        self.dPos = ps.PosStable(alpha, beta, generator = self.gen)
        self.rho = self.dPos.rho
        self.theta = self.dPos.theta
        self.ES_1_a = self.dPos.mellin(19*alpha/20)
        self.calc_params()
        self.ES_1_gamma = self.dPos.mellin(self.gamma)
        
    def params(self):
        return (self.alpha, self.beta, self.theta, self.rho)
    
    def all_params(self):
        return (self.alpha, self.beta, self.theta, self.rho, self.d, self.delta, self.gamma, self.kappa, self.Delta_0, self.m_star, self.eta)

    def minimum(self):
        return 0 if self.beta == -1 and self.alpha <= 1 else 1 if self.alpha == 1 else 0
    
    def maximum(self):
        return 0 if self.beta == -1 and self.alpha <= 1 else 1 if self.alpha == 1 else np.inf
    
    def insupport(self, x):
        return x == 0 if self.beta == -1 and self.alpha <= 1 else x >= 0
    
    def mean(self):
        return self.alpha*self.rho*self.dPos.mean()

    def calc_cdf1(self): #to save time during first iteration of algorithm_3
        n = -self.Delta_0-1
        n += 1
        aux = np.exp(self.delta*n)
        self.cdf1 = self.dPos.cdf(aux)

    def calc_eta(self):
        alpha = self.alpha
        rho = self.rho
        d = self.d
        eta = - alpha*rho - lambertw(-alpha*rho*d*np.exp(-alpha*rho*d), -1)/d
        if np.imag(eta) != 0:
            print('Error: Imaginary eta')
        else:
            self.eta = np.real(eta)

    def set_params(self, omega, eta = None):
        #6-tuple of parameters omega
        (self.d, self.delta, self.gamma, self.kappa, self.Delta_0, self.m_star) = omega
        if eta == None:
            self.calc_eta()
        else:
            self.eta = eta
        assert 0 < self.delta < self.d < 1/(self.alpha*self.rho)
        assert self.kappa >= 0
        assert 0 < self.gamma < self.alpha
        assert self.Delta_0 < 0 , 'Delta_0 = '+str(self.Delta_0)
        assert isinstance(self.Delta_0, int) , 'Delta_0 = '+str(self.Delta_0)
        self.calc_cdf1()

    def calc_params(self):
        alpha = self.alpha
        rho = self.rho
        self.d = 2/(3*alpha*rho)
        self.calc_eta()
        self.delta = 1/(3*alpha*rho)
        self.gamma = (19/20)*alpha
        self.kappa = 4 + max(alpha*rho*np.log(2)/(self.eta*2), 1/(alpha*rho))
        self.Delta_0 = -40
        temp = np.floor((60/19)*rho*np.log(self.ES_1_a))
        self.m_star = 12 + (temp if temp > 0 else 0)
        self.calc_cdf1()

    def print_params(self):
        #code to print alpha, beta, rho, delta, gamma, kappa, Delta_0, m_star)
        print('alpha: ', self.alpha)
        print('beta: ', self.beta)
        print('rho: ', self.rho)
        print('delta: ', self.delta)
        print('gamma: ', self.gamma)    
        print('kappa: ', self.kappa)
        print('Delta_0: ', self.Delta_0)
        print('m_star: ', self.m_star)
        print('eta: ', self.eta)
    
    def func_a(self, theta):
        #theta = (s, u, w, lambda)
        (s, u, w, lam) = theta
        alpha = self.alpha
        return (1/lam-1)*pow((1-u)/u,1/alpha)*s

    def func_psi(self, x, theta):
        (s, u, w, lam) = theta
        alpha = self.alpha
        rho = self.rho
        t1 = np.power(w,1/(alpha*rho))*np.power(1-u,1/alpha)*s
        t2 = lam*(np.power(u,1/alpha)*x + pow(1-u,1/alpha)*s)
        return t1 if self.func_a(theta) >= x else t2

    def algorithm_3(self, conditional, shift = None): #shift = m-k >= m_star - 1
        n = self.m_star - 1 if shift == None else shift
        gen = self.gen
        U = gen.uniform()
        S = []
        while True:
            n = n + 1
            aux = np.exp(self.delta*n)
            aux0 = np.exp(self.delta*(n+1))
            cdf1 = self.cdf1 if not conditional else self.dPos.cdf(aux) #conditional tells whether it's the first iteration of alg_9
            if conditional:
                cdf0 = self.dPos.cdf(aux0)
                if cdf0 == 0:
                    p = 1
                else:
                    p = cdf1/cdf0
            else:
                p = cdf1
            q_factor = np.exp(-(n+1)*self.delta*self.gamma)*self.ES_1_gamma #aux1
            q = p*np.exp((-1/(1-np.exp(-self.delta*self.gamma)))*q_factor/(1-q_factor)) 
            if U > p:
                S_temp = self.dPos.rv() #aux2
                while S_temp[0] < aux or (conditional and S_temp[0] > aux0):
                    S_temp = self.dPos.rv()
                S = np.concatenate((S, S_temp))
                U = (U - p)/(1 - p)
            elif U < q:
                S_temp = self.dPos.rv()
                while S_temp[0] > aux:
                    S_temp = self.dPos.rv()
                S = np.concatenate((S, S_temp))
                return S
            else:
                S_temp = self.dPos.rv()
                while S_temp[0] > aux:
                    S_temp = self.dPos.rv()
                S = np.concatenate((S, S_temp))
                U = U/p
    
    def sample_C(self, x):
        #returns C = [C_0, C_-1, C_-2, ..., C_Tx]
        C = [0]
        gen = self.gen
        if x > 0:
            t = 0
            while True:
                if C[t] - C[0] > x:
                    return np.array(C)
                t += 1
                F = self.d - gen.exponential()/(self.alpha*self.rho+self.eta)
                C.append(C[t-1] + F)
        else:
            t = 0
            while True:
                if C[t] - C[0] < x:
                    return np.array(C)
                t += 1
                F = self.d - gen.exponential()/(self.alpha*self.rho)
                C.append(C[t-1] + F)
    
    def calc_T_y(self, C, y):
        if y > 0:
            for t in range(len(C)):
                if C[t] - C[0] > y:
                    return -t
            return -len(C)
        else:
            for t in range(len(C)):
                if C[t] - C[0] < y:
                    return -t
            return -len(C)
        
    def calc_L(self, C, x):
        y = 0
        while True:
            T_y = self.calc_T_y(C, y)
            if T_y > x:
                return y
            y += 1

    def calculate_R(self, C, n, Delta_t):
        #calculate R_i for i = Delta_t, Delta_t+1, ..., n-1; R_i = R[-i]; R_end[0] = R_(n-1), R_end[i] = R_{n-1-i}; R_n = max C_k - C_n for end <= k <= n
        R_end = np.zeros(n-Delta_t)
        for ii in range(n-1, Delta_t, -1):
            R_end[n-1-ii] = np.max(C[-ii:]) - C[-n]
        return R_end

    def algorithm_4(self, x, x_prime):
        assert x_prime > x > 0, "algorithm_4 : x = "+str(x)+" x_prime = "+str(x_prime)+"\n"
        m = np.exp(-x_prime-self.d)
        gen = self.gen
        while True:
            T = -np.log(gen.uniform(m, 1))/self.eta
            if T <= x:
                return 0
            C = self.sample_C(T)
            max = np.max(C[:-1])
            if max <= x_prime:
                return 0 if max <= x else 1

    def algorithm_5(self, x, x_prime):
        assert x_prime > x > 0, "x ="+str(x)+" x_prime ="+str(x_prime)
        gen = self.gen
        while True:
            C = self.sample_C(x)
            U = gen.uniform()
            if x_prime == np.inf:
                ind = 1
            else:
                ind = 1 - self.algorithm_4(x_prime - C[-1], np.inf)
            if U <= np.exp(-self.eta*C[-1]) and ind == 1:
                return C
            
    def algorithm_6(self, x, x_prime):
        assert 0 < x < np.inf and 0 < x_prime, 'x = '+str(x)+' x_prime = '+str(x_prime)
        while True:
            C = self.sample_C(-x)
            if x_prime == np.inf:
                ind = 1
            else:
                ind = 1-self.algorithm_4(x_prime - C[-1], np.inf)
            if ind == 1 and max(C) <= x_prime:
                return C
    
    def algorithm_8(self, F):
        alpha, rho, d = self.alpha, self.rho, self.d
        U, Lambda = [], []
        gen = self.gen
        for k in range(len(F)):
            T = 1 + gen.poisson(-alpha*(F[k]-d)*(1-rho))
            if T - 1 == 0:
                U.append(np.exp(alpha*(F[k]-d)))
                Lambda.append(1)
            else:
                L = gen.beta(1, T-1)
                U.append(np.exp(alpha*(F[k]-d)*L))
                Lambda.append(np.exp((1-L)*alpha*(F[k]-d)))        
        return (U, Lambda)

    def algorithm_9(self):
        #Delta[n] = Delta_n; C[n] = C_{-n}; U, S, Lambda[-n-1] = U_{n}, S_{n}, Lambda_{n}
        Delta = [self.Delta_0]
        x, t, s, m, n = np.inf, 0, Delta[0], Delta[0] + 1, Delta[0] + 1
        gen = self.gen
        U = gen.uniform(size = -Delta[0])
        W = gen.uniform(size = -Delta[0])
        Lambda = 1 + gen.binomial(1, 1-self.rho, size = -Delta[0]) * (np.power(gen.uniform(size = -Delta[0]), 1/(self.alpha * self.rho)) - 1 )
        S = self.dPos.rv(size = -Delta[0])
        lag_U = gen.uniform(size = -Delta[0])
        lag_W = gen.uniform(size = -Delta[0])
        lag_Lambda = 1 + gen.binomial(1, 1-self.rho, size = -Delta[0]) * (np.power(gen.uniform(size = -Delta[0]), 1/(self.alpha * self.rho)) - 1 )
        lag_S = self.dPos.rv(size = -Delta[0])
        C = np.array([0])
        R = np.array([0])
        counta = 0
        while True:
            counta += 1
            S_end = self.algorithm_3(conditional = True if counta != 1 else False, shift=-s-1)
            S = np.append(S, S_end)
            chi = -len(S) 
            countb = 0
            while n == m or Delta[t] > chi:
                countb += 1
                if len(C) < -Delta[t]:
                    while len(C) < -Delta[t]:
                        C = np.append(C, C[-1] + self.d - gen.exponential()/(self.alpha*self.rho+self.eta))
                C_end = C[-1] + self.algorithm_6(x = 2*self.kappa, x_prime = x-C[-1])[1:]
                C = np.concatenate((C, C_end))
                Delta.append(-(len(C)-1))
                t += 1
                ind_R = self.algorithm_4(x=self.kappa, x_prime=x - C[-1])
                if ind_R == 0:
                    while len(R) < -n + 1:
                        R = np.append(R, None)
                    R_end = self.calculate_R(C, n, Delta[t-1])
                    R = np.concatenate((R, R_end))
                    n = Delta[t-1]
                    x = C[-1] + self.kappa
                else:
                    C_end = C[-1]+self.algorithm_5(x = self.kappa, x_prime = x - C[-1])[1:]
                    C = np.concatenate((C, C_end))
                    Delta.append(-(len(C)-1))
                    t += 1
            F_end = [*range(s, chi, -1)]               #F_{n} = F_end[-(n-s)]
            for i in range(s, chi, -1):
                F_end[-(i-s)] = C[-i] - C[-(i+1)]   #F_i = C_i - C_{i+1}
            U_end, Lambda_end = self.algorithm_8(F_end)
            W_end = gen.uniform(size = np.size(U_end))   
            U = np.concatenate((U, U_end))
            Lambda = np.concatenate((Lambda, Lambda_end))
            W = np.concatenate((W, W_end))
            s = chi
            D_sum = np.sum(np.exp(-self.d*(n-1-np.arange(chi, m-1)))*(S[-(m-1):-chi][::-1])*np.power(1-(U[-(m-1):-chi][::-1]), 1/self.alpha))
            D = np.exp(R[-m+1])*(np.exp((self.d-self.delta)*(chi-(m-1)))/(1-np.exp(self.delta - self.d)) + D_sum)
            m = m - 1
            if D <= self.func_a(theta = (S[-m-1], U[-m-1], W[-m-1], Lambda[-m-1])):
                X_0 = D
                for i in range(m, 0):
                    X_0 = self.func_psi(X_0, theta = (S[-i-1], U[-i-1], W[-i-1], Lambda[-i-1]))
                for i in range(Delta[0]):
                    X_0 = self.func_psi(X_0, theta = (lag_S[i], lag_U[i], lag_W[i], lag_Lambda[i]))
                return X_0
            elif m == Delta[0]:
                coalescense = False
                X_0 = D
                for i in range(m, 0):
                    X_0 = self.func_psi(X_0, theta = (S[-i-1], U[-i-1], W[-i-1], Lambda[-i-1]))
                    if X_0 <= self.func_a(theta = (S[-i-2], U[-i-2], W[-i-2], Lambda[-i-2])):
                        coalescense = True
                if coalescense:
                    for i in range(Delta[0]):
                        X_0 = self.func_psi(X_0, theta = (lag_S[i], lag_U[i], lag_W[i], lag_Lambda[i]))
                    return X_0
                
    def rv(self, size=1, show_progress = False):
        sample = np.zeros(size)
        i = 0
        while i < size:
            if i%10 == 0 and show_progress:
                print(i)
            sample[i] = self.algorithm_9()
            i += 1
        return sample

from unilateralstable import UniStable
from stable import Stable
from positivestable import PosStable
