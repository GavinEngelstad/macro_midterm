from helpers import macro_modeling as mm
import matplotlib.pyplot as plt
import numpy as np

'''
Love you Mario, but please start using an IDE

Get steady state, check it, get matricies, and run impulse
response.

The first few lines show the steady state values. The next
show the calculated equations using the steady states. If
the calculated steady state is right, they equation should
all be 0 (plus/minus some machine error).

Parameters can all be updated to fit.
'''

# parameters
z_ss = 1
beta = 0.9
nu = 0.85
phi = 0.5
delta = 0.1
alpha = 0.36
rho = 0.95
sim_t = 100

## Helpful functions
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
                [0, C_ss],
                [0, 0]])
    B = np.array([[alpha, 0],
                [-1, 0],
                [0, 0],
                [-(1-delta)*K_ss, 0],
                [0, 0],
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
    J = np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 0, beta*lambda_ss*r_ss, 0, beta*lambda_ss*(r_ss+1-delta)]])
    K = np.array([[0, 0, 0, 0, 0, -lambda_ss],
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

# solve it
solution_res = solve(z_ss, beta, nu, phi, delta, alpha, rho)

# get steady state values
z_ss, Y_ss, K_ss = solution_res['z_ss'], solution_res['Y_ss'], solution_res['K_ss']
N_ss, C_ss, I_ss = solution_res['N_ss'], solution_res['C_ss'], solution_res['I_ss']
r_ss, w_ss, lambda_ss = solution_res['r_ss'], solution_res['w_ss'], solution_res['lambda_ss']

# Steady state values
print(f'z_ss = {z_ss}')
print(f'Y_ss = {Y_ss}')
print(f'K_ss = {K_ss}')
print(f'N_ss = {N_ss}')
print(f'C_ss = {C_ss}')
print(f'I_ss = {I_ss}')
print(f'r_ss = {r_ss}')
print(f'w_ss = {w_ss}')
print(f'lambda_ss = {lambda_ss}')
print()

# Check Steady State
print('1.', z_ss * K_ss**alpha * N_ss**(1-alpha) - Y_ss)
print('2.', alpha*Y_ss*K_ss**(-1)-r_ss)
print('3.', (1-alpha)*Y_ss*N_ss**(-1)-w_ss)
print('4.', K_ss-(1-delta)*K_ss-I_ss)
print('5.', Y_ss - C_ss - I_ss)
print('6.', 1/(C_ss-nu*C_ss)-beta*nu/(C_ss-nu*C_ss)-lambda_ss)
print('7.', lambda_ss*w_ss-phi)
print('8.', beta*(lambda_ss*r_ss+(1-delta)*lambda_ss)-lambda_ss)
print()

# matricies
print(f'P = {solution_res['P']}')
print(f'Q = {solution_res['Q']}')
print(f'R = {solution_res['R']}')
print(f'S = {solution_res['S']}')

# Run it
ir_res = impulse_response(solution_res, sim_t)

# Plot it
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)
ax.set_ylabel(r'\% Deviation from Steady State')
ax.set_xlabel('Time Since Shock')

ax.plot(ir_res['T'], ir_res['zhat_t'], label=r'$\hat{z}_t$')
ax.plot(ir_res['T'], ir_res['Yhat_t'], label=r'$\hat{Y}_t$')
ax.plot(ir_res['T'], ir_res['Khat_t'], label=r'$\hat{K}_t$')
ax.plot(ir_res['T'], ir_res['Chat_t'], label=r'$\hat{C}_t$')
ax.plot(ir_res['T'], ir_res['Ihat_t'], label=r'$\hat{I}_t$')
ax.plot(ir_res['T'], ir_res['Nhat_t'], label=r'$\hat{N}_t$')
ax.plot(ir_res['T'], ir_res['rhat_t'], label=r'$\hat{r}_t$')
ax.plot(ir_res['T'], ir_res['what_t'], label=r'$\hat{w}_t$')
ax.plot(ir_res['T'], ir_res['lambdahat_t'], label=r'$\hat{\lambda}_t$')

ax.legend()

plt.show()