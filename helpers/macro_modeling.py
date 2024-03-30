import numpy as np
import numpy.polynomial.polynomial as POLY
from scipy import linalg


""" CONTENTS (as of 02/13/2024):
    1. Jacobian (self-explanatory; input for Newton solver)
    
    2. Newton (self-explanatory, tool for solving nonlinear systems of 
               equations)
    
    3. uc_xyz (finds policy functions; divides model variables into 
       endogenous states, exogenous states, and other endogenous vars)
    
    4. impulse_xyz (generates impulse-response functions using output of 
       uc_system_xyz)
    
    5. simulate_xyz (generates one draw of simulated model data using
       output of uc_system_xyz)
    
    6. uc_xz (finds policy functions; brute-force version of uc_xyz
       that divides model variables into endogenous and exogenous
       vars)
    
    7. impulse_xz (generates impulse-response functions using output of 
       uc_xz)
    
    8. simulate_xz (generates one draw of simulated model data using
       output of uc_xz)
    
    STILL TO ADD:
    9. statistics (generates standard deviations and correlations using 
                   simulated data) """




def Jacobian(function_name, X, parameter_vector, step_size):
    """ This program calculates the Jacobian matrix of a vector-valued function 
    F(X). The inputs of the program are a (vector-valued) function 
    (function_name), a vector X in R**N (X), a parameter vector 
    (parameter_vector), and a step size h (step_size). """

    # Find the dimensions of matrix J and initialize
    FX = eval('function_name(X, parameter_vector)')
    rows, R1 = np.shape(FX)          
    columns, C1 = np.shape(X)
    J = np.zeros((rows, columns))        

    # Building the step size matrix
    H = step_size * max(1, np.amax(abs(X)))
    Q = np.identity(columns)            
    H *= Q

    # Fill out the entries in J
    for k in range(0, columns):
        J[:, k:(k+1)] = ((eval('function_name(X+H[:, k:(k+1)], parameter_vector)') 
                        - eval('function_name(X, parameter_vector)')) / 
                        np.linalg.norm(H, np.inf))
    
    return(J)




def Newton(function_name, X0, parameter_vector, tolerance):
    """ This function calculates the zero of any vector-valued function F(x).
    The inputs are a vector-valued function F (function_name), an initial guess 
    X0 in R**N (X0), a parameter vector in R**K (parameter_vector), and a 
    tolerance value h. """

    # Initialize the value of the function at X0
    FX = eval('function_name(X0, parameter_vector)')

    # Loop until convergence
    while np.linalg.norm(FX) > tolerance:
        J = eval('Jacobian(function_name, X0, parameter_vector, tolerance)')
        FX = eval('function_name(X0, parameter_vector)')
        X0 = X0 - np.matmul(np.linalg.inv(J), FX)

    return X0




def uc_xyz(A, B, C, D, F, G, H, J, K, L, M, N):
    """This function implements the undetermined coefficients solution method 
    as outlined by Uhlig (1999). It divides the model variables into three
    vectors, namely
    
    x: endogenous state variables
    y: endogenous (control) variables
    z: exogenous state variables. """

    rF, m = np.shape(F) 
    rN, k = np.shape(N)
    Psi = F - np.matmul(J,np.linalg.solve(C,A))
    Gamma = (np.matmul(J,np.linalg.solve(C,B)) 
                - G + np.matmul(K,np.linalg.solve(C,A)))
    Theta = np.matmul(K,np.linalg.solve(C,B)) - H

    # This first section builds the matrix P, considering the actual size of 
    # the state variable vector

    if m == 1:
        # If there is only one state variable, I can improve the speed by 
        # solving P as the root of a quadratic equation
        PC0 = [-Theta[0, 0] , -Gamma[0, 0] , Psi[0, 0]]
        PC1 = POLY.polyroots(PC0)
        PC2 = abs(PC1)
        P = np.atleast_2d(min(PC2))

    else:
        # When there is more than one state variable, I need to make some
        # changes relative to the calculations above; in particular, I need to
        # solve for a matrix quadratic equation
        I_m = np.identity(m)
        O_m = np.zeros((m, m))
        Xi = np.block([[Gamma, Theta], [I_m, O_m]])
        Delta = np.block([[Psi, O_m], [O_m, I_m]])

        # Generalized eigenvalues and eigenvectors
        # Use "from scipy import linalg" at the top of the script
        eigval,eigvec = linalg.eig(Xi, Delta)
        eigval = np.real_if_close(eigval)
        rv, cv = np.shape(eigvec)

        # Transforming eigenvector so it fits Matlab's equivalent
        for i in range(cv):
            eigvec[:, i] = eigvec[:, i]/(np.amax(abs(eigvec[:, i])))

        # Choose the stable eigenvalues and corresponding eigenvectors 
        index = np.where(eigval<=1)
        ZZ = [index for index, value in enumerate(eigval) if abs(value) <= 1]
        Phi1 = np.diag(eigval[ZZ])
        X12 = eigvec[rv - len(ZZ):rv, ZZ]

        # Obtain matrix P
        P = np.matmul(X12, np.matmul(Phi1, np.linalg.inv(X12)))
        P = np.real_if_close(P)

    # This last part builds matrices Q, R, and S
    R = np.linalg.solve(-C, np.matmul(A, P) + B)
    Ik = np.identity(k)
    W = (np.kron(np.transpose(N), F - np.matmul(J, np.linalg.solve(C, A)))
            + np.kron(Ik, np.matmul(J, R) + np.matmul(F, P) + G 
            - np.matmul(K, np.linalg.solve(C, A))))
    Q0 = (np.matmul(np.matmul(J, np.linalg.solve(C, D)) - L, N)
            + np.matmul(K, np.linalg.solve(C, D)) - M)
    Q1 = np.reshape(Q0, (rF*k, 1))
    Q2 = np.linalg.solve(W, Q1)
    Q = np.reshape(Q2, (rF, k))
    S = np.linalg.solve(-C, np.matmul(A, Q)+D)

    return(P,Q,R,S)




def impulse_xyz(P, Q, R, S, N, shocked_variable, number_of_periods, sigma_z = 0.01):
    """This function creates impulse-response series for vectors x and y  
    given a time-zero sigma_z shock to variable shocked_variable of vector  
    z (where the first shock has index 0) following the state-space system
    
    x_(t+1) = Px_t + Qz_t
        y_t = Rx_t + Sz_t
        z_t = Nz_(t+1) + e_t.
    
    Matrices [P, Q, R, S, N] and a number_of_periods horizon are given. """
    
    # Preliminaries: these operations get the dimensions of vectors x and y
    rows_x, cols_x = np.shape(P)
    rows_y, cols_y = np.shape(S)
    
    # Create the impulse-response data
    
    # Shock vector z
    z = np.zeros((1, number_of_periods + 1))
    z[0, 0] = sigma_z
    for j in range(1, number_of_periods + 1):
        z[0, j] = N[shocked_variable, shocked_variable] * z[0, j-1]
    
    # State variables
    if cols_x == 1:
        x = np.zeros((cols_x, 1))
            # Starting the iteration at period 0

        for colx in range(number_of_periods):
            x_stacked = P * x[0, colx] + Q[0, shocked_variable] * z[0, colx]
            x = np.hstack((x, x_stacked))

    else:
        x = np.zeros((cols_x, 1))

        for colx in range(number_of_periods):
            x_stacked = (np.matmul(P, x[:, colx]) 
                + Q[:, shocked_variable] * z[0, colx])
            x_stacked = np.transpose(np.atleast_2d(x_stacked))
            x = np.hstack((x, x_stacked))
     
    # Endogenous variables
    
    if (cols_y == 1 and cols_x == 1):
        y = R * x[0, 0] + S * z[0, 0]
            # Starting the iteration at period 0

        for coly in range(number_of_periods):
            y_stacked = R * x[0, coly] + S * z[0, coly]
            y = np.hstack((y, y_stacked))  

    else:
        y = (np.matmul(R, x[:, 0]) 
             + S[:, shocked_variable] * z[0, 0])
        y = np.transpose(np.atleast_2d(y))
        # Starting the iteration at period 0

        for coly in range(number_of_periods):
            y_stacked = (np.matmul(R, x[:, coly]) 
                         + S[:, shocked_variable] * z[0, coly])
            y_stacked = np.transpose(np.atleast_2d(y_stacked))
            y = np.hstack((y, y_stacked))
  
    x *= 100
    y *= 100
    z *= 100
    
    return(x, y, z)




def simulate_xyz(P, Q, R, S, N, number_of_periods, SIGMA):
    """This function creates simulated series for vectors x_t and y_t given a
    stochastic shock process z_t following the state-space system
    
    x_(t+1) = Px_t + Qz_t
        y_t = Rx_t + Sz_t
        z_t = Nz_(t-1) + e_t.
    
    Matrices [P, Q, R, S, N] and a number_of_periods horizon are given, and e_t 
    is an iid Gaussian disturbance with mean zero and variance-covariance matrix
    SIGMA*SIGMA'. """

    # Here I increase T by one unit to include period 0
    T = number_of_periods 
    T += 1 
    
    # Preliminaries: these operations get the dimensions of vectors x, y, and z
    rows_x, cols_x = np.shape(P)
    rows_y, cols_y = np.shape(S)
    rows_z, cols_z = np.shape(N)
     
    # Create the simulated data   
    eps = np.random.default_rng().normal(0, 1, size=(cols_z, T))
    eps *= SIGMA

    z = np.zeros((cols_z, T))

    for cz in range(2,T):
        z[:, cz] = np.matmul(N, z[:, cz - 1]) + eps[:, cz]

    # Create the state variables time path
    x = np.zeros((cols_x, T))

    for cx in range(1,T-1):
        x[:, cx + 1] = np.matmul(P, x[:, cx]) + np.matmul(Q, z[:, cx]);
        # Recall that the policy function for endogenous states takes the
        # form 
        #
        # x_(t+1) = Px_t + Qz_t

    # Create the endogenous variables time path
    y = np.matmul(R, x) + np.matmul(S, z)
    # Recall that the policy function is of the form 
    #
    # y_t = Rx_t + Sz_t

    x *= 100
    y *= 100
    z *= 100
    
    return(x, y, z)




def uc_xz(F, G, H, L, M, N):
    """This function implements the undetermined coefficients solution method 
    as outlined by Uhlig (1999). It divides the model variables into two
    vectors, namely
    
    x: endogenous (state and control) variables
    z: exogenous variables. """

    rF, m = np.shape(F) 
    rN, k = np.shape(N)
    Psi = F
    Gamma = - G
    Theta = - H

    # This first section builds the matrix P, considering the actual size of 
    # the state variable vector

    if m == 1:
        # If there is only one state variable, I can improve the speed by 
        # solving P as the root of a quadratic equation
        PC0 = [-Theta[0, 0] , -Gamma[0, 0] , Psi[0, 0]]
        PC1 = POLY.polyroots(PC0)
        PC2 = abs(PC1)
        P = np.atleast_2d(min(PC2))

    else:
        # When there is more than one state variable, I need to make some
        # changes relative to the calculations above; in particular, I need to
        # solve for a matrix quadratic equation
        I_m = np.identity(m)
        O_m = np.zeros((m, m))
        Xi = np.block([[Gamma, Theta], [I_m, O_m]])
        Delta = np.block([[Psi, O_m], [O_m, I_m]])

        # Generalized eigenvalues and eigenvectors
        eigval, eigvec = linalg.eig(Xi, Delta)
        eigval = np.real_if_close(eigval)
        rv, cv = np.shape(eigvec)

        # Transforming eigenvector so it fits Matlab's equivalent
        for i in range(cv):
            eigvec[:, i] = eigvec[:, i]/(np.amax(abs(eigvec[:, i])))

        # Choose the stable eigenvalues and corresponding eigenvectors 
        index = np.where(eigval<=1)
        ZZ = [index for index, value in enumerate(eigval) if abs(value) <= 1]
        Phi1 = np.diag(eigval[ZZ])
        X12 = eigvec[rv-len(ZZ):rv, ZZ]

        # Obtain matrix P
        P = np.matmul(X12, np.matmul(Phi1, np.linalg.inv(X12)))
        P = np.real_if_close(P)

    # This last part builds matrix Q
    FP = np.matmul(F, P)
    FPG = FP + G
    Ik = np.identity(k)
    V = np.kron(Ik, FPG) + np.kron(np.transpose(N), F)
    LN = np.matmul(L, N)
    LNM = LN + M
    Q1 = -np.reshape(LNM, (rF * k, 1))
    Q2 = np.linalg.solve(V, Q1)
    Q = np.reshape(Q2, (rF, k))

    return(P, Q)




def impulse_xz(P, Q, N, shocked_variable, number_of_periods, sigma_z = 0.01):
    """This function creates impulse-response series for vectors x and y  
    given a time-zero sigma_z shock to variable shocked_variable of vector  
    z (where the first shock has index 0) following the state-space system
    
        x_t = Px_(t-1) + Qz_t
        z_t = Nz_(t-1) + e_t.
    
    Matrices [P, Q, N] and a number_of_periods horizon are given. """
    
    # Preliminaries: these operations get the dimensions of vector x
    rows_x, cols_x = np.shape(P)
    
    # Create the impulse-response data
    
    # Shock vector z
    z = np.zeros((1, number_of_periods + 1))
    z[0, 0] = sigma_z
    for j in range(1, number_of_periods + 1):
        z[0, j] = N[shocked_variable, shocked_variable] * z[0, j-1]
    
    # Endogenous variables
    x = np.zeros((cols_x, 1))

    for colx in range(number_of_periods):
        x_stacked = (np.matmul(P, x[:, colx]) 
            + Q[:, shocked_variable] * z[0, colx])
        x_stacked = np.transpose(np.atleast_2d(x_stacked))
        x = np.hstack((x, x_stacked))
    
    x *= 100
    z *= 100
    
    return(x, z)




def simulate_xz(P, Q, N, number_of_periods, SIGMA):
    """This function creates simulated series for vector x_t given a
    stochastic shock process z_t following the state-space system
    
        x_t = Px_(t-1) + Qz_t
        z_t = Nz_(t-1) + e_t.
    
    Matrices [P, Q, N] and a number_of_periods horizon are given, and e_t 
    is an iid Gaussian disturbance with mean zero and variance-covariance 
    matrix SIGMA*SIGMA'. """

    # Increase T by one unit to include period 0
    T = number_of_periods
    T += 1 
    
    # These operations get the dimensions of vectors x and z
    rows_x, cols_x = np.shape(P)
    rows_z, cols_z = np.shape(N)
     
    # Create the simulated data   
    eps = np.random.default_rng().normal(0, 1, size=(cols_z, T))
    eps *= SIGMA

    z = np.zeros((cols_z, T))
    
    for cz in range(1,T):
        z[:, cz] = np.matmul(N, z[:, cz - 1]) + eps[:, cz]
    
    # Create the state variables time path
    x = np.zeros((cols_x, T))
    
    for cx in range(1,T):
        x[:, cx] = np.matmul(P, x[:, cx - 1]) + np.matmul(Q, z[:, cx])
        # Recall that the policy function for endogenous states takes the
        # form 
        #
        # x_t = Px_(t-1) + Qz_t

    x *= 100
    z *= 100
    
    return(x, z)



