from helpers import macro_modeling as mm
import numpy as np

def solve(z_ss,
          beta,
          nu,
          phi,
          delta,
          alpha,
          rho
          ):
    ## Steady States
    z_ss = 1
    r_ss = 1/beta - 1 + delta
    K_ss = z_ss**(1/(1-alpha))*(1-alpha)*(1 - beta*nu)/(phi*(1-nu)*(r_ss/alpha-delta))*(r_ss/alpha)**(alpha/(alpha-1))
    Y_ss = r_ss/alpha * K_ss
    C_ss = (r_ss/alpha - delta) * K_ss
    I_ss = delta * K_ss
    N_ss = (r_ss/(alpha*z_ss))**(1/(1-alpha)) * K_ss
    w_ss = phi*(1-nu)/(1-beta*nu)*(r_ss/alpha-delta) * K_ss
    lambda_ss = (1-beta*nu)/((1-nu)*C_ss)

    ## Matricies
    # matrix equation 1 (known)
    A = np.array([[0, 0],
                [0, 0],
                [0, 0],
                [K_ss, 0],
                [0, 0],
                [0, 0]])
    B = np.array([[alpha, 0],
                [-1, 0],
                [0, 0],
                [-(1-delta)*K_ss, 0],
                [0, C_ss],
                [0, 0]])
    C = np.array([[-1, 0, (1-alpha), 0, 0, 0],
                [1, 0, 0, -1, 0, 0],
                [1, 0, -1, 0, -1, 0],
                [0, -I_ss, 0, 0, 0, 0],
                [-Y_ss, I_ss, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1]])
    D = np.transpose([[1, 0, 0, 0, 0, 0]])

    # matrix equation 2 (expectations)
    F = np.array([[0, beta*nu/((1-nu)**2*C_ss)],
                [0, 0]])
    G = np.array([[0, -(1+beta*nu**2)/((1-nu)**2*C_ss)],
                [0, 0]])
    H = np.array([[0, nu/((1-nu)**2*C_ss)],
                [0, 0]])
    J = np.array([[0, 0, 0, 0, 0, -lambda_ss],
                [0, 0, 0, beta*lambda_ss*r_ss, 0, beta*lambda_ss*(r_ss+1-delta)]])
    K = np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -lambda_ss]])
    L = np.zeros((2, 1))
    M = np.zeros((2, 1))

    # matrix equation 3 (law of motion for stochastic variables)
    N = np.array([[rho]])

    ## Get policy function
    P, Q, R, S = mm.uc_xyz(A, B, C, D, F, G, H, J, K, L, M, N)

    return {
        'z_ss': z_ss,
        'Y_ss': Y_ss,
        'K_ss': K_ss,
        'C_ss': C_ss,
        'I_ss': I_ss,
        'N_ss': N_ss,
        'r_ss': r_ss,
        'w_ss': w_ss,
        'lambda_ss': lambda_ss,
        'P': P,
        'Q': Q,
        'R': R,
        'S': S,
        'N': N
    }


def impulse_response(res, sim_t):
    # get matricies
    P, Q, R, S, N = res['P'], res['Q'], res['R'], res['S'], res['N']

    # simulate it
    T = np.arange(sim_t+1)
    res = mm.impulse_xyz(P, Q, R, S, N, 0, sim_t)
    Khat_t, Chat_t = res[0]
    Yhat_t, Ihat_t, Nhat_t, rhat_t, what_t, lambdahat_t = res[1]
    zhat_t = res[2][0,:]

    return {
        'T': T,
        'zhat_t': zhat_t,
        'Yhat_t': Yhat_t,
        'Khat_t': Khat_t,
        'Chat_t': Chat_t,
        'Ihat_t': Ihat_t,
        'Nhat_t': Nhat_t,
        'rhat_t': rhat_t,
        'what_t': what_t,
        'lambdahat_t': lambdahat_t
    }


def pmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX pmatrix as a string

    I copied this from the internet, all credit to 
    https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """
    if len(a.shape) > 2:
        raise ValueError('pmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{pmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{pmatrix}']
    return '\n'.join(rv)