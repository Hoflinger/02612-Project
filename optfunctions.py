import numpy as np
from scipy import linalg, sparse
from sksparse import cholmod
import time

# Optimization toolkit for course 02612


#build kkt system for system in form:
#   min 0.5x'Hx + g'x
#   st  A'x = b
def kkt(H,g,A,b):
    m = A.shape[1]
    Z = np.zeros((m,m))
    if sparse.issparse(H):
        KKT = sparse.bmat([[H,-A],[-A.T,None]])
    else:
        KKT = np.block([[H,-A],[-A.T,Z]])
        
    return KKT, np.concatenate((-g,-b))


# ----- Equality constrained convex QP solvers -----

# ----- Dense solvers -----

def EqualityQPSolver_LUdense(H,g,A,b):
    KKT, RHS = kkt(H,g,A,b)
    lu_factor = linalg.lu_factor(KKT)
    sol = linalg.lu_solve(lu_factor,RHS)
    x, lambdas = sol[:len(g)], sol[len(g):]
    return x, lambdas

def EqualityQPSolver_LDLdense(H,g,A,b):
    KKT, RHS = kkt(H,g,A,b)
    lu,d,perm = linalg.ldl(KKT, lower=True)
    y = linalg.solve(d, linalg.solve_triangular(lu[perm], RHS[perm], lower=True))
    sol = np.zeros(len(RHS))
    sol[perm] = linalg.solve_triangular(lu[perm], y, trans=1, lower=True)
    x, lambdas = sol[:len(g)], sol[len(g):]
    return x, lambdas

def EqualityQPSolver_NullSpace(H,g,A,b):
    q,r = linalg.qr(A)
    m1 = r.shape[1]
    Q1,Q2,R = q[:,:m1], q[:,m1:], r[:m1,:m1]
    xY = linalg.solve_triangular(R, b, trans=1)
    xZ = linalg.solve(Q2.T @ H @ Q2, -Q2.T @ (H @ Q1 @ xY + g))
    x = Q1 @ xY + Q2 @ xZ
    lambdas = linalg.solve_triangular(R, Q1.T @ (H @ x + g))
    return x, lambdas

def EqualityQPSolver_RangeSpace(H,g,A,b):
    H_fac = linalg.cho_factor(H)
    v = linalg.cho_solve(H_fac, g)

    H_inv = linalg.cho_solve(H_fac,np.eye(len(g)))
    HA = A.T @ H_inv @ A
    HA_fac = linalg.cho_factor(HA)

    lambdas = linalg.cho_solve(HA_fac, b + A.T @ v)
    
    x = linalg.cho_solve(H_fac, A @ lambdas - g)
    return x, lambdas

# ----- Sparse solvers -----

def EqualityQPSolver_LUsparse(H,g,A,b):
    KKT, RHS = kkt(H,g,A,b)
    KKT = sparse.csc_matrix(KKT)
    B = sparse.linalg.splu(KKT)
    sol = B.solve(RHS)
    x, lambdas = sol[:len(g)], sol[len(g):]
    return x, lambdas

#Note for sparse ldl solver:
#does not do bunch kaufman factorization -> positive definite kkt matrix needed!
def EqualityQPSolver_LDLsparse(H,g,A,b):
    KKT, RHS = kkt(H,g,A,b)
    KKT = sparse.csc_matrix(KKT)
    factor = cholmod.cholesky(KKT)
    sol = factor.solve_LDLt(RHS)
    x, lambdas = sol[:len(g)], sol[len(g):]
    return x, lambdas

#Interface to all solvers:
def EqualityQPSolver(H,g,A,b,solver):
    '''
    Equality constrained convex QP solver of form:
        min x'Hx + gx
        st. A'x = b
    where H is symmetric and positive-definite
    Params:
        H:      H matrix from obj function
        g:      g vector from obj function
        A:      A matrix from constraints
        b:      b vector from constraints
        solver: string to indicate which solver:
            "LUdense":      for dense system using LU factorization
            "LUsparse":     for sparse system using LU factorization
            "LDLdense":     for dense system using LDL factorization
            "LDLsparse":    for sparse system using LDL factorization
            "NullSpace":    for dense system using the Null-Space method
            "RangeSpace":   for dense system using the Range-Space/ Schur-Complement method

    Output:
        x:          solution to minimisation problem
        lambda_:    the associated lagrange multipliers
    '''
    if solver.lower() == "ludense":
        return EqualityQPSolver_LUdense(H,g,A,b)
    elif solver.lower() == "lusparse":
        return EqualityQPSolver_LUsparse(H,g,A,b)
    elif solver.lower() == "ldldense":
        return EqualityQPSolver_LDLdense(H,g,A,b)
    elif solver.lower() == "ldlsparse":
        return EqualityQPSolver_LDLsparse(H,g,A,b)
    elif solver.lower() == "nullspace":
        return EqualityQPSolver_NullSpace(H,g,A,b)
    elif solver.lower() == "rangespace":
        return EqualityQPSolver_RangeSpace(H,g,A,b)
    
    raise NameError(f"no solver named: {solver}. check spelling.")


def convert_to_standard(l, u, C, dl, du):
    '''
    converts problem:
        min 0.5x'Hx + g'x
        st. l <= x <= u
           dl <= C'x <= du

    to form:
        min 0.5x'Hx + g'x
        st. A'x >= b
    '''
    A = np.block([[np.eye(len(l))],[-np.eye(len(l))],[C.T],[-C.T]])
    b = np.concatenate((l,-u,dl,-du))
    return A.T, b


#Convert problem to big M form for active set method
def bigM_form(H,g,A,b,M):
    n, m = A.shape
    H = np.block([[H,np.zeros(n)[:,None]],[np.zeros(n+1)]])
    H[-1,-1] = 1
    g = np.concatenate((g,[M]))
    sys = np.block([[A,np.zeros(n)[:,None]],[np.ones(m+1)]])
    return H, g, sys, np.concatenate((b,[0]))



#Active set algorithm as given in the slides:
def CQP_solver_active(H ,g , A, b, x0, max_iter = 100, tol=1e-8, solver="LUdense", verbose=False, save_it=False):
    """
    Active Set algorithm for convex QPs of form:

        min 0.5x'Hx + g'x
        st. A'x >= b

    ---------------------------------------------

    parameters:
    ___________
        H : H matrix of convex obj function
        g : g vector of convex obj function
        A : Constraints matrix
        b : contstraints RHS
        x0 : starting point
        max_iter : optional max iterations
    ___________
    output:
    ___________
        x : minimum of CQP
        lambdas : lambdas of constraints
        active_set : active set of constraints
        k : iterations needed to solve problem
        converged: bool if converged or not
    """

    converged = False
    indx = np.arange(len(b))                #index list for constraints
    ws = np.zeros(len(b)).astype(np.bool_)  #boolean list working set

    xks = np.zeros((max_iter,len(x0)))      #store x iterations
    wss = np.zeros((max_iter,len(ws)))      #store working set iterations
    lambdass = np.zeros((max_iter,len(b)))  #store lagrange multiplier iterations

    if np.any(A.T @ x0 < b):
        raise ValueError("Initial point is not feasible!")

    x = x0

    for i in range(max_iter):

        xks[i] = x

        #define working sets
        Aw, bw = A[:,ws], b[ws]

        #solve for search direction pk
        pk, lambdass[i,ws] = EqualityQPSolver(H, (H @ x + g), Aw, np.zeros(len(bw)), solver=solver)
        #pk = sol[:len(x0)]

        #if |pk| is zero
        if np.allclose(pk,0, rtol=tol):
            #lambdass[i,ws] = sol[len(x0):] 

            #check stopping criteria for lagrange multipliers
            if np.all(lambdass[i] >= 0):
                converged = True
                break
            else:
                #exit constraint with most negative lagrange multiplier
                j = np.argmin(lambdass[i])
                ws[j] = False
                if verbose: print(f"exiting constraint: {j}")

        #if |pk| is non-zero
        else:

            #calculate maximum step-size
            alphas = np.zeros(len(b))
            m = (A.T @ pk < 0) & ~ws
            alphas[m] = (b[m] - A.T[m] @ x) / (A.T[m] @ pk)
            alpha = min(1,np.min(alphas[m]))
            #update x
            x = x + alpha*pk

            #enter blocking constraint into working set
            if alpha < 1:
                blocking = np.argmin(alphas[m])
                block_indx = indx[m][blocking]
                if verbose: print(f"entering constraint: {block_indx}")
                ws[block_indx] = True

    return x, xks[:i+1], lambdass[:i+1], wss[:i+1], i, converged



#algorithm using big M for active set method:
def active_set(H, g, C, dl, du, l, u, max_iter=1000,  tol=1e-8, solver="LUdense", verbose=False):
    '''
    solve convex QP using active set method with big M

    '''
    #convert to form st. A'x >= b
    A, b = convert_to_standard(l,u,C,dl,du)

    #Convert problem to big M form to find initial feasible point:
    #M is chosen small in the start, if t is non zero at final solution,
    # M is increased and the problem is resolved until t is 0.
    M = 2
    times = []
    t_is_0 = False
    time_active = 0
    t1_tot = time.perf_counter()
    while not t_is_0:

        #initial point heuristics
        Ha, ga, Aa, ba = bigM_form(H,g,A,b,M)
        x0 = np.concatenate((u - l, [0]))
        t = np.max(Aa.T @ x0)
        x0[-1] = t

        #run active set algorithm
        t1 = time.perf_counter()
        sol_active, xks, lambdaks, wks, iter_active, converged = CQP_solver_active(Ha,ga,Aa,ba,x0, 
                                                                                   max_iter=max_iter, 
                                                                                   tol=tol,
                                                                                   solver=solver,
                                                                                   verbose=verbose)
        t2 = time.perf_counter()

        if np.allclose(sol_active[-1],0):
            t_is_0 = True
            time_active = t2-t1
            print(f"solved successfully with M = {M}")
            print(f"    solved in {time_active}s,")
            print(f"    and {iter_active+1} iterations.")
        else:
            M = M*2
    t2_tot = time.perf_counter()
    time_active_total = t2_tot - t1_tot
    print(f"Total time: {time_active_total}s")

    return sol_active, xks, lambdaks, wks, iter_active, converged




#primal-dual interior point algorithm from slides

#initial point heuristics for primal dual interior point algorithm
def primal_dual_initial(H,g,A,b):
    n, m = A.shape
    x_bar = np.zeros(n)
    z_bar = np.ones(m)/m
    s_bar = np.ones(m)/m

    rL = H @ x_bar + g - A @ z_bar
    rA = s_bar + b - A.T @ x_bar
    rSZ = s_bar * z_bar

    H_bar = H + (A * (z_bar / s_bar)) @ A.T
    lu_fac = linalg.lu_factor(H_bar)

    rL_bar = rL - (A * (z_bar / s_bar)) @ (rA - rSZ/z_bar)

    dx_aff = linalg.lu_solve(lu_fac, -rL_bar)
    
    dz_aff = - ((z_bar / s_bar)[:,None] * A.T) @ dx_aff + (z_bar / s_bar)*(rA - rSZ / z_bar)
    ds_aff = - rSZ / z_bar - s_bar / z_bar * dz_aff

    z = np.maximum(1,np.abs(z_bar + dz_aff))
    s = np.maximum(1,np.abs(s_bar + ds_aff))

    return x_bar, z, s

#algorithm based on Mehrotra's predictor-corrector method
def primal_dual_interior(H,g,l,u,C,dl,du, x0=None, z0=None, s0=None, max_iter=100, tol=10e-8):
    '''
    Primal-dual interior point algorithm for convex QPs of form:

        min 0.5x'Hx + g'x
        st. A'x >= b

    ---------------------------------------------

    parameters:
    ___________
        H : H matrix of convex obj function
        g : g vector of convex obj function
        A : Constraints matrix
        b : contstraints RHS
        x0 : starting point
        z0 : initial lagrange multipliers
        s0 : intital slack variables
        max_iter : optional max iterations
        tol : tolerance for stopping criteria
    ___________
    output:
    ___________
        x : optimal x
        xks : x iterations
        iter : iteration at convergence
        converged: bool if converged or not 
    '''

    A, b = convert_to_standard(l,u,C,dl,du)

    n, m = A.shape

    x0, z0, s0 = primal_dual_initial(H,g,A,b)

    xks = np.zeros((max_iter+2,n))
    zks = np.zeros((max_iter+2,m))
    sks = np.zeros((max_iter+2,m))
    res = np.zeros((max_iter+2,3))

    x = x0
    z = z0
    s = s0

    z_ratio = np.ones(m)
    s_ratio = np.ones(m)

    rL = H @ x + g - A @ z
    rA = s + b - A.T @ x
    rSZ = s * z
    mu = np.dot(s,z)/m

    converged = False
    iter = 0

    rL_cond = tol * max(1, H.max(), g.max(), A.max())
    rA_cond = tol * max(1, b.max(), A.max())
    mu_cond = tol * 10e-2 * mu
    #print(f"rL_cond: {rL_cond}")
    #print(f"rA_cond: {rA_cond}")
    #print(f"mu_cond: {mu_cond}")

    while not converged and iter < max_iter:
        #print(f"iteration: {iter}\n")
        xks[iter] = x
        zks[iter] = z
        sks[iter] = s
        res[iter] = np.array([abs(rL.max()),abs(rA.max()),mu])

        H_bar = H + (A * (z / s)) @ A.T
        lu_fac = linalg.lu_factor(H_bar)

        # affine step
        rL_bar = rL - (A * (z / s)) @ (rA - rSZ/z)
        dx_aff = linalg.lu_solve(lu_fac, -rL_bar)

        dz_aff = - ((z / s)[:,None] * A.T) @ dx_aff + (z / s) * (rA - rSZ / z)
        ds_aff = - (rSZ / z) - s / z * dz_aff
  
        z_ratio_aff = np.where(-z/dz_aff <= 0, 1, -z/dz_aff)
        s_ratio_aff = np.where(-s/ds_aff <= 0, 1, -s/ds_aff)
        alpha_aff = min(1, z_ratio_aff.min(), s_ratio_aff.min())

        mu_aff = (z + alpha_aff * dz_aff).T @ (s + alpha_aff * ds_aff) / m
        sigma = (mu_aff / mu) ** 3

        rSZ_bar = rSZ + ds_aff * dz_aff - sigma * mu * np.ones(m)
        rL_bar = rL - (A * (z / s)) @ (rA - rSZ_bar/z)
        
        dx = linalg.lu_solve(lu_fac, -rL_bar)
        dz = - ((z / s)[:,None] * A.T) @ dx + (z / s)*(rA - rSZ_bar / z)
        ds = - rSZ_bar / z - s / z * dz

        z_ratio = np.where(-z/dz <= 0, 1, -z/dz)
        s_ratio = np.where(-s/ds <= 0, 1, -s/ds)
        alpha = min(1, z_ratio.min(), s_ratio.min())

        eta = 0.995
        alpha_bar = eta * alpha

        x = x + alpha_bar * dx
        z = z + alpha_bar * dz
        s = s + alpha_bar * ds

        rL = H @ x + g - A @ z
        rA = s + b - A.T @ x
        rSZ = s * z
        mu = np.dot(s,z)/m

        iter += 1

        if abs(rL.max()) <= rL_cond and abs(rA.max()) <= rA_cond and mu <= mu_cond:
            converged = True
            xks[iter] = x
            zks[iter] = z
            sks[iter] = s    
            res[iter] = np.array([abs(rL.max()),abs(rA.max()),mu])
    
    return x, xks[:iter+1], zks[:iter+1], sks[:iter+1], res[:iter+1], iter, converged
