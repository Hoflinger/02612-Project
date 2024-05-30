#Linear programming tools:
import numpy as np
from scipy import linalg, sparse
import time


#convert lp to standard form
def lp_standard_form(g, A, b, l=None, u=None):
    '''
    Transform problem of form:
        min g'x
        st Ax=b
        l<=x<=u
    To form:
        min g'x
        st Ax=b
           x>=0
    '''

    m, n = A.shape
    x_eye = np.eye(n)

    #if no lower or upper bounds:
    if l is None and u is None:
        return np.concatenate((g, -g)), np.hstack((A,-A)), b

    #incase lower or upper bounds not supplied:
    if l is None:
        l = np.full(n, -np.inf)
    if u is None:
        u = np.full(n, np.inf)

    #check for lower bound == upper bound:
    lbequb = l == u
    if np.any(lbequb):
        constr = np.eye(n)[lbequb]
        rhs = l[lbequb]
        A = np.vstack((A, constr))
        b = np.concatenate((b, rhs))
        l = l[~lbequb]
        u = u[~lbequb]

    #check for nonnegativity on x, otherwise x = x^+ - x^-
    if not np.all(l >= 0):
        A = np.hstack((A,-A))
        x_eye = np.hstack((x_eye, -x_eye))

    #indexes to include in new constraints matrix:
    l_idx = np.logical_and(np.isfinite(l), l!=0)
    u_idx = np.isfinite(u)

    if np.any(l_idx):
        print(A.shape)
        print(np.eye(n)[l_idx,l_idx])
        A_new = np.block([[A,              np.zeros((m,n))[:,l_idx],           np.zeros((m,n))[:,u_idx]            ],
                          [x_eye[l_idx], -np.eye(n)[np.ix_(l_idx,l_idx)],      np.zeros((n,n))[np.ix_(l_idx,u_idx)]],
                          [x_eye[u_idx], np.zeros((n,n))[np.ix_(u_idx,l_idx)], np.eye(n)[np.ix_(u_idx,u_idx)]      ]])

        b_new = np.concatenate((b,l[l_idx],u[u_idx]))

        g_new = np.concatenate((g, -g, np.zeros(l_idx.sum() + u_idx.sum())))

    else:
        A_new = np.block([[A,            np.zeros((m,n))[:,u_idx]],
                          [x_eye[u_idx], np.eye(n)[u_idx]        ]])

        b_new = np.concatenate((b,l[l_idx],u[u_idx]))

        g_new = np.concatenate((g, np.zeros(l_idx.sum() + u_idx.sum())))

    return g_new, A_new.squeeze(), b_new


#Phase 1 simplex form
def lp_simplex_phase1_form(A, b):
    '''
    Linear program for phase 1 simplex.
    takes an LP of form:
        min g'x
        st. Ax=b
            x>=0
    
    and converts it to:
        min t
        st. [A  I] [x t]' = b

            [x t]'>= 0
    
    initial feasible points are given by:
    x0 = 0
    t_i = |b_i|
    '''

    m, n = A.shape

    diag = np.sign(b)
    diag[diag == 0] = 1

    A_init = np.hstack((A, np.diag(diag)))

    x0_init = np.concatenate((np.zeros(n), np.abs(b)))
    g_init = np.concatenate((np.zeros(n), np.ones(m)))

    n_nonbasic = A_init.shape[1] - A_init.shape[0]
    idx_list = np.arange(A_init.shape[1])
    Bindex = idx_list[n_nonbasic:]
    Nindex = idx_list[:n_nonbasic]

    return g_init, A_init, b, x0_init, Bindex, Nindex


#Revised Simplex algorithm
def lp_simplex_alg(g, A, b, x0, Bindex, Nindex, max_iter=100, verbose=False):
    '''
    simplex algorithm on matrix form:
    assumes standard form structure, ie.
            min g'x
        st. Ax = b
            x >= 0
    where g -> R^n, A -> R^mxn, b -> R^m

    also assumes knowledge of basic and nonbasic vars.

    params:
        g:          vector, linear objective coefficients
        A:          Equality constraint coeffs
        b:          Equality constraints rhs

    returns:
        x:          optimal solution
        basic:      basic variables at optimum
        non_basic:  non basic variables at optimum
        iter:       iterations

    '''
    is_sparse = sparse.issparse(A)

    n_basic = A.shape[0]
    n_nonbasic = A.shape[1] - n_basic

    vars = np.zeros(len(g))


    xB = x0[Bindex]
    xN = x0[Nindex]

    STOP = False
    iter = 0
    
    while not STOP and iter < max_iter:
        if verbose: print(f"iteration {iter}.\n")
        B = A[:,Bindex]
        N = A[:,Nindex]

        lu_factor = linalg.lu_factor(B)
        mu = linalg.lu_solve(lu_factor, g[Bindex], trans=1)
        lambdaN = g[Nindex] - N.T @ mu

        if np.all(lambdaN >= 0 - 1e-15):
            vars[Bindex], vars[Nindex] = xB, xN
            return vars, Bindex, Nindex, iter
        
        s = np.argmin(lambdaN)

        sindx = Nindex[s]

        h = linalg.lu_solve(lu_factor, A[:,sindx]) 

        if np.all(h <= 0):
            print("unbounded solution")
            return
        
        ratios = np.ones(n_basic)*100000
        ratios[h > 0] = xB[h > 0] / h[h > 0]
        j = np.argmin(ratios)

        alpha = xB[j] / h[j]

        xB = xB - alpha * h
        xB[j] = alpha
        xN[s] = 0
        Bindex[j], Nindex[s] = Nindex[s], Bindex[j]
        
        iter += 1

        if verbose:
            print(f"xB: {xB}")       
            print(f"Basic indexes: {Bindex}")
            print(f"entering index s: {s}")
            print(f"most negative lambda: {lambdaN.min()}")
            print(f"leaving index j: {j}")
            print(f"ratio: {ratios.min()}")

    if not STOP:
        print("Max iterations reached, consider increasing max_iter!")
        return


#Simplex procedure (conversion of problem form, phase1, phase2)
def lp_simplex(gx, Aeq, beq, lb, ub, max_iter=100, verbose=False, run_phase1=False):
    '''
    revised simplex algorithm according to slides provided:
    The program converts linear programs of form
        min g'x
        st. A'x=b
            l<=x<=u
    into:
        min g'x
        st. Ax=b
            x>=0
    
    introducing slack variables where necessary and 
    then solves the LP using the revised simplex method.
    If an initial point is not easily found, perform 
    phase 1 of the simplex algorithm to find initial
    feasible point. Thereafter uses initial feasible 
    point to solve phase 2 of simplex. 

    params:
        gx:         objective coefficients
        Aeq:        equality constraints matrix
        beq:        rhs of equality constraints
        lb:         lower bound on original x
        ub:         upper bound on original x
        max_iter:   (optional) maximum iterations
        verbose:    (optional) if true, print info of problem
                    at each iteration
        run_phase1: (optional) if true then phase1 is forced to run

    returns:
        sol:    returns solution to phase 1 if infeasible,
                else returns solution phase 2.
        iter:   returns num iterations in phase 1 if infeasible,
                else returns num iterations in phase 2.
    '''
    print("Attempting to solve LP using revised simplex algorithm:")
    t1_t = time.perf_counter()
    n_vars, n_constraints = Aeq.shape

    #transpose A:
    Aeq = Aeq.T

    #convert to standard form:
    g, A, b = lp_standard_form(gx, Aeq, beq, lb, ub)

    m, n = A.shape
    n_nonbasic = n - m

    #bypass phase 1 if initial point is easily found
    #by setting slack variables = b:
    bypass = False
    idxlist = np.arange(n) #index list for variables

    #assign slack variables as basic and x variables as nonbasic
    Bindex = idxlist[n_nonbasic:]
    Nindex = idxlist[:n_nonbasic]

    x0 = np.zeros(n)
    x0[Bindex] = b

    #if easy guess of x0 is feasible, then bypass phase1
    if np.all(A @ x0 == b) and not run_phase1:
        bypass = True

    #phase1 of simplex
    if not bypass:
        print("Solving phase 1 simplex:")
        t1_p1 = time.perf_counter()
        g_init, A_init, b_init, x0_init, Bindex_init, Nindex_init = lp_simplex_phase1_form(A, b)
        sol_p1, Bix_p1, Nix_p1, iter_p1 = lp_simplex_alg(g_init, A_init, b_init, 
                                                        x0=x0_init, Bindex=Bindex_init, Nindex=Nindex_init, 
                                                        max_iter=max_iter, verbose=verbose)
        t2_p1 = time.perf_counter()

        obj_p1 = g_init.T @ sol_p1
        print(f"Phase 1 objective: {obj_p1}")
        #if objective value of phase1 is not 0, LP is infeasible
        if not np.allclose(obj_p1, 0):
            print("Infeasible problem.")
            return sol_p1, iter_p1

        #if feasible, use solution for phase1 as intial point in phase2
        Bindex = Bix_p1[Bix_p1 < n]
        Nindex = Nix_p1[Nix_p1 < n]
        x0 = sol_p1[:n]

        print(f"solved phase 1 in {t2_p1-t1_p1}s and {iter_p1} iterations.")

    else:
        print("phase 1 of simplex has been bypassed.")

    if verbose: print(f"initial feasible point:\n{x0}")
    
    #phase2 of simplex
    t1_p2 = time.perf_counter()

    sol_p2, Bix_p2, Nix_p2, iter_p2 = lp_simplex_alg(g, A, b, x0=x0, 
                                                     Bindex=Bindex, Nindex=Nindex, 
                                                     max_iter=max_iter, verbose=verbose)
    t2_p2 = time.perf_counter()

    x = sol_p2[:n_vars]
    t2_t = time.perf_counter()
    print(f"solved phase 2 in {t2_p2-t1_p2}s and {iter_p2} iterations.")
    print("optimal x:")
    print(x)
    print(f"optimal objective: {gx.T @ x}")
    print(f"total run time: {t2_t - t1_t}")


    return x, iter_p2


#Initial feasible point for primal-dual interior-point algorithm
def lp_ip_init(g, A, b):
    '''
    initial point heuristics for interior point LP of form:
        min g'x
        st. Ax=b
            x>=0
    with lagrangian:
        L(x,y,z) = g'x - y'(Ax-b) - z
    params:
        g:  n dimensional objective coefficients
        A:  m x n dimensional constraints matrix
        b:  n dimensional constraints rhs

    returns:
        x0: initial x
        y0: initial y
        z0: initial z
    '''
    x_bar = A.T @ linalg.inv(A @ A.T) @ b
    y_bar = linalg.inv(A @ A.T) @ A @ g
    z_bar = g - A.T @ y_bar

    x_hat = x_bar + np.max(-1.5*np.min(x_bar),0) 
    z_hat = z_bar + np.max(-1.5*np.min(z_bar),0)

    x0 = x_hat + 0.5 * np.dot(x_hat,z_hat) / np.sum(z_hat)
    z0 = z_hat + 0.5 * np.dot(x_hat,z_hat) / np.sum(x_hat)

    return x0, y_bar, z0


#primal-dual interior-point algorithm for standard form:
#THIS CODE WORKS
def lp_ip(gx, Aeq, beq, lb, ub, max_iter=1000, tol=1e-8, verbose=False):
    '''
    solve LP of form:
        min g'x
        st. Ax = b
            x >= 0
    Params:
        g:  n dimensional objective coefficients
        A:  m x n dimensional Constraints matrix
        b:  m dimensional constraints rhs
    optional:
        max_iter:   int, max iterations
        tol:        float, stopping criteria tolerance
        verbose:    bool, true prints several parameter values for trouble shooting
    '''

    print("Solving LP using interior-point predictor-corrector method:")
    t1 = time.perf_counter()

    g, A, b = lp_standard_form(gx, Aeq.T, beq, lb, ub)

    m, n = A.shape

    x0, y0, z0 = lp_ip_init(g, A, b)

    xks = np.zeros((max_iter+2,n))
    yks = np.zeros((max_iter+2,m))
    zks = np.zeros((max_iter+2,n))
    residuals = np.zeros((max_iter+2,3))

    x = x0
    y = y0
    z = z0

    #define intitial residuals:
    rL = g - A.T @ y - z #dual feasibility
    rA = - A @ x + b #primal feasibility
    rXZ = x * z #complementary conditions
    s = np.dot(x, z) / n #duality gap

    eta = 0.995

    STOP = False
    iter = 0

    while not STOP and iter < max_iter:
        
        #save iterations to memory
        xks[iter] = x
        yks[iter] = y
        zks[iter] = z
        residuals[iter] = np.array([np.linalg.norm(rL,2),
                                    np.linalg.norm(rA,2),
                                    np.linalg.norm(rXZ,2)])

        #cholesky factorize matrix (A Z^-1 X A') 
        mat = (A * (x / z)) @ A.T
        factor = linalg.cho_factor(mat)

        #solve system (AZ^-1XA')dy_aff = rA + A(Z^-1 X rL + Z^-1 rXZ) 
        rA_bar = rA + A @ ( rL * x / z + rXZ / z)
        dy_aff = linalg.cho_solve(factor, rA_bar)

        #back substitute to find dx_aff, dz_aff
        dx_aff = (A.T * (x / z)[:,None]) @ dy_aff - x / z * rL - rXZ / z
        dz_aff = -rXZ / x - z / x * dx_aff

        #calculate max step sizes alpha and beta for primal 
        #and dual spaces according to 14.32 and 14.33 in 
        #Numerical Optimization J. Nocedal, S. J. Wright
        alpha_aff = min(1, np.min(-x[dx_aff < 0]/dx_aff[dx_aff < 0])) 
        beta_aff = min(1, np.min(-z[dz_aff < 0]/dz_aff[dz_aff < 0]))

        #duality gap for affine step
        s_aff = np.dot(x + alpha_aff * dx_aff, z + beta_aff * dz_aff) / n
        #centering parameter
        sigma = (s_aff / s) ** 3

        #solve for rhs of aggregated predictor, corrector and centering contributions
        rXZ_bar =  rXZ + dx_aff * dz_aff - sigma * s * np.ones(n)  
        rA_bar = rA + A @ ( rL * x / z + rXZ_bar / z)
        dy = linalg.cho_solve(factor, rA_bar)

        #back substitute for dx and dy
        dx = (A.T * (x / z)[:,None]) @ dy - x / z * rL - rXZ_bar / z
        dz = -rXZ_bar / x - z / x * dx

        #calculate step sizes 
        alpha = min(1, np.min(-x[dx < 0]/dx[dx < 0])) 
        beta = min(1, np.min(-z[dz < 0]/dz[dz < 0]))

        #update parameters
        x = x + eta * alpha * dx
        y = y + eta * beta * dy
        z = z + eta * beta * dz

        #update residuals
        rL = g - A.T @ y - z
        rA = - A @ x + b
        rXZ = x * z
        s = np.dot(x, z) / n

        if verbose:
            print(f"iter: {iter}:\n")
            print(f"s: {s}")
            print(f"|rL|: {linalg.norm(rL,2)}")
            print(f"|rA|: {linalg.norm(rA,2)}")
            print(f"alpha: {alpha}, beta: {beta}")
            print(f"complementarity: {np.allclose(x*z, 0)}")
            print(f"obj: {g.T @ x}\n")
            
        iter += 1
        
        #check convergence criteria: ||rL|| < tol, ||rA|| < tol, |s| < tol
        if linalg.norm(rL,2) <= tol and linalg.norm(rA,2) <= tol and abs(s) <= tol:
            xks[iter] = x
            yks[iter] = y
            zks[iter] = z
            residuals[iter] = np.array([np.linalg.norm(rL,2),
                                    np.linalg.norm(rA,2),
                                    np.linalg.norm(rXZ,2)])
            STOP = True
    
    t2 = time.perf_counter()

    print(f"Optimal x: \n{x[:n]}")
    print(f"Objective value: {g.T @ x[:n]}")
    print(f"Solved in {t2-t1}s and {iter} iterations.")
            
    return x, xks[:iter], yks[:iter], zks[:iter], residuals[:iter], iter, STOP



#primal-dual interior-point algorithm for LP with bounded x
#THIS CODE DOES NOT WORK
def lp_ip_bounded(g, A, b, l, u, x0=None, max_iter=100, tol=1e-8, verbose=False):
    '''
    Primal-dual nterior-point predictor-corrector algorithm for linear programs (LPs) of form:
            min g'x
        st. A'x = b
        l <= x <= u
    with lagrangian:
        L(x,l,y,z,t) = g'x - l'(Ax-b) - y'x - 

    references:
    [2] Gondzio, J. “Multiple centrality corrections in a primal dual method for linear programming.” 
        Computational Optimization and Applications, Volume 6, Number 2, 1996, pp. 137–156.

    Params:
        g: n dimensional vector of objective coefficients
        A: n x m dimensional array of equality constraints
        b: m dimensional vector of rhs of equality constraints
        l: n dimensional vector of lower bound for x
        u: n dimensional vector of upper bound for x
    '''
    n = len(g)
    m = len(b)
    

    print(n)
    print(m)

    #ensure A is of shape n x m incase only 1 equality constraint:
    if len(A.shape) == 1:
        A = A[:,np.newaxis]

    
    #transform bound constraint to be of form 0 <= x <= u_bar:
    #let x_bar = x - l:, then new upper bound = u - l:
    u_bar = u - l
    b_bar = b - A.T @ l

    #declare intital variables
    if x0 is None:
        x0 = np.ones(n) * u_bar / 2 #x variables
        l0 = np.zeros(m)            #lagrange multipliers for eq constraints: l'(A'x-b)
        y0 = np.ones(n)             #non-negativity constraint on x: y_i*x_i=0, i=1,2,...,n
        z0 = np.ones(n)             #non-negativity constraint on t: z_i*t_i=0, i=1,2,...,n
        t0 = np.ones(n)             #slack variable for upper bound constraint: x+t=d
    
    x0, l0, y0, z0, t0 = lp_ip_init(g, A, b_bar, u_bar)

    CONVERGED = False
    iter = 0

    xks = np.zeros((max_iter+2,n))
    lks = np.zeros((max_iter+2,m))
    yks = np.zeros((max_iter+2,n))
    zks = np.zeros((max_iter+2,n))
    tks = np.zeros((max_iter+2,n))

    x = x0
    l = l0
    y = y0
    z = z0
    t = t0

    rL = g - A @ l - y - z  #dual feasibility
    rA = - A.T @ x + b_bar      #primal feasibility
    rUB = x + t - u_bar     #primal feasibility
    rYX = y*x               #nonnegativity on x
    rZT = z*t               #nonnegativity on slack variable t
    mu = (np.dot(x,y) + np.dot(z,t)) / (2 * n)
    
    #stop cond:
    #TODO: subject to change
    rho = max(1, A.max(), g.max(), b_bar.max())
    rL_cond = tol * rho
    rA_cond = tol * rho
    rUB_cond = tol * rho
    mu_cond = tol * 10e-2 * mu

    eta = 0.995

    for iter in range(max_iter+1):
        if verbose:
            print(f"---------------\nITERATION: {iter}\n")
            print("residuals:")
            print(f"rL: {linalg.norm(rL,2)}\nrA: {linalg.norm(rA,2)}\nrUB: {linalg.norm(rUB,2)}\nmu1: {mu}")

        xks[iter] = x
        lks[iter] = l
        yks[iter] = y
        zks[iter] = z
        tks[iter] = t

        #check stopping criteria:
        if linalg.norm(rL,2) <= rL_cond and linalg.norm(rA,2) <= rA_cond and linalg.norm(rUB,2) <= rUB_cond and mu <= mu_cond :
            CONVERGED = True
            break

        D = (y / x - z / t)
        r = - rL - rZT / t - rYX / x + (z / t) * rUB 

        #affine direction:
        ADA =  A.T @ ((1/D)[:,np.newaxis] * A)
        if m > 1:
            cho_fac = linalg.cho_factor(ADA)
            dl_aff = linalg.cho_solve(cho_fac, - rA - (A / D[:,np.newaxis]).T @ r)
            
        #if only one equality constraint, 
        else: 
            dl_aff = (- rA - ((A / D[:,np.newaxis]).T @ r)) / ADA.squeeze()
        
        dx_aff = (r + A @ dl_aff) / D
        dt_aff = - rUB - dx_aff
        dy_aff = - rYX / x - (y / x) * dx_aff 
        dz_aff = - rZT / t - (z / t) * dt_aff

        alpha_aff = min(1, np.min(-x[dx_aff < 0]/dx_aff[dx_aff < 0], initial=1),
                        np.min(-t[dt_aff < 0]/dt_aff[dt_aff < 0], initial=1))
        beta_aff = min(1, np.min(-z[dz_aff < 0]/dz_aff[dz_aff < 0], initial=1),
                        np.min(-y[dy_aff < 0]/dy_aff[dy_aff < 0],initial=1))

        mu_aff = np.dot(x + alpha_aff * dx_aff, y + beta_aff * dy_aff) + np.dot(t + alpha_aff * dt_aff, z + beta_aff * dz_aff)
        #centering parameter according to [2]
        sigma = ((mu_aff / (2 * n)) / mu) ** 2 * (mu_aff / n)

        #affine centering correction direction:
        rYX_bar = rYX + dx_aff * dy_aff - sigma 
        rZT_bar = rZT + dt_aff * dz_aff - sigma
        r_bar = - rL - rZT_bar / t - rYX_bar / x + rUB / t

        if m > 1:
            cho_fac = linalg.cho_factor(ADA)
            dl = linalg.cho_solve(cho_fac, - rA - (A / D[:,np.newaxis]).T @ r_bar)

        #if only one equality constraint, 
        else: 
            dl = (- rA - ((A / D[:,np.newaxis]).T @ r_bar)) / ADA.squeeze()

        dx = (r_bar + A @ dl_aff) / D
        dt = - rUB - dx
        dy = - rYX_bar / x - (y / x) * dx
        dz = - rZT_bar / t - (z / t) * dt

        alpha = min(1, np.min(-x[dx < 0]/dx[dx < 0], initial=1),
                        np.min(-t[dt < 0]/dt[dt < 0], initial=1))
        beta = min(1, np.min(-z[dz < 0]/dz[dz < 0], initial=1),
                        np.min(-y[dy < 0]/dy[dy < 0],initial=1))

        #update iteration:
        x += alpha * eta * dx
        l += beta * eta * dl
        y += beta * eta * dy
        z += beta * eta * dz
        t += alpha * eta * dt

        #compute residuals:
        rL = g - A @ l - y - z
        rA = - A.T @ x + b_bar
        rUB = x + t - u_bar
        rYX = y * x
        rZT = z * t 
        mu = (np.dot(x,y) + np.dot(z,t)) / (2 * n)
        #mu1 = np.dot(x,y) / n
        #mu2 = np.dot(z,t) / n 

        if verbose:
            print("----STEPSIZES----")
            print(f"alpha_aff: {alpha_aff}\nbeta_aff: {beta_aff}")
            print(f"alpha: {alpha}\nbeta: {beta}")
            print(f"obj: {g.T @ (x+l)}")
    
    #back transform x
    x = x + l
    xks = xks + l

    #save other vars as 
    vars = {"l": l, "y": y, "z": z, "t": t}
    saved_iters = {"lks": lks, "yks": yks, "zks": zks, "tks": tks}

    obj = g.T @ x
    return x, obj, iter, CONVERGED, vars, saved_iters 


