from os import name
from tracemalloc import start
from scipy.sparse import spdiags
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import solve, lu, solve_triangular
from scipy.sparse.linalg import bicgstab, gmres
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# linear operator for finite difference method
# matrix A: Laplacian operator
L = 10
m = 64
n = m * m 
span = [-L, L]
dx = 2*L / m
dy = dx
e0 = np.zeros((n, 1)) 
e1 = np.ones((n, 1)) 
e2 = np.copy(e1) 
e4 = np.copy(e0) 
for j in range(1, m+1):
    e2[m*j-1] = 0 
    e4[m*j-1] = 1 
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
matA = spdiags(diagonals, offsets, n, n).toarray()
matA /= dx**2

# matrix B: partial x operator
e1 = np.ones((n, 1))
e2 = -np.ones((n, 1))
e3 = np.copy(e2)
e4 = np.copy(e1)
diagonals = [e1.flatten(), e2.flatten(), e3.flatten(), e4.flatten()]
offsets = [m, -m, n-m, -(n-m)]
matB = spdiags(diagonals, offsets, n, n).toarray()
matB /= (2*dx)

# matrix C: partial y operator
e1 = np.ones((n, 1))
e2 = -np.ones((n, 1))
e3 = np.zeros((n, 1))
e4 = np.zeros((n, 1))
for j in range(1, m+1):
    e1[m*(j-1)] = 0
    e2[m*j-1] = 0
    e3[m*j-1] = -1 
    e4[m*(j-1)] = 1 
diagonals = [e1.flatten(), e2.flatten(), e3.flatten(), e4.flatten()]
offsets = [1, -1, m-1, -(m-1)]
matC = spdiags(diagonals, offsets, n, n).toarray()
matC /= (2*dy)

# for gmres residuals calculation
wk = 0

def fft_stream(w, KX, KY, K, n, N, A, L, U, P):
    w = w[:N].reshape((n, n))
    w_t = fft2(w)
    psi_t = -w_t / K
    psi = np.real(ifft2(psi_t))
    return psi.flatten(), False

def direct_stream(w, KX, KY, K, n, N, A, L, U, P):
    psi = solve(A, w)
    return psi.flatten(), False

def lu_stream(w, KX, KY, K, n, N, A, L, U, P):
    y = solve_triangular(L, np.dot(P, w), lower=True)
    psi = solve_triangular(U, y)
    return psi.flatten(), False

residuals_gm = []
def bicgstab_callback(xk):
    residuals_gm.append(np.linalg.norm(abs(matA.dot(xk) - wk)))
    
def bicgstab_stream(w, KX, KY, K, n, N, A, L, U, P):
    xk = w
    psi, exitcode = bicgstab(A, w, atol=1e-6, callback = bicgstab_callback)
    return psi.flatten(), True

def gmres_callback(pr_norm):
    residuals_gm.append(pr_norm)
    
def gmres_stream(w, KX, KY, K, n, N, A, L, U, P):
    xk = w
    psi, exicode = gmres(A, w, atol=1e-6, callback=gmres_callback)
    return psi.flatten(), True
    
def vorticity_rhs(_, w, nu, n, N, solve_stream, A, B, C, KX, KY, K, L, U, P):
    psi, has_residuals = solve_stream(w, KX, KY, K, n, N, A, L, U, P)
    if has_residuals:
        if solve_stream.__name__ == "gmres_stream":
            print(f"GMRES converged in {len(residuals_gm)} iterations with residual {residuals_gm[-1]}")
        else:
            print(f"BICGSTAB converged in {len(residuals_gm)} iterations with residual {residuals_gm[-1]}")
        residuals_gm.clear()
    w_t = fft2(w.reshape((n, n)))
    return -B.dot(psi) * C.dot(w) + C.dot(psi) * B.dot(w) + nu * (A.dot(w))

def main():
    # different methods to solve the stream function 
    n = 64
    N = n**2
    L = 20
    nu = 0.001
    tp = [0, 4]
    tspan = np.arange(tp[0], tp[1] + 0.5, 0.5)
    xspan = np.linspace(span[0], span[1], n+1)
    yspan = np.copy(xspan)
    x = xspan[:n]
    y = yspan[:n]
    [X, Y] = np.meshgrid(x, y)
    w0 = np.exp(-X**2 - Y**2 / 20).flatten()

    kx = (2 * np.pi / L) * np.concatenate((np.arange(0, n//2), np.arange(-n//2, 0)))
    ky = (2 * np.pi / L) * np.concatenate((np.arange(0, n//2), np.arange(-n//2, 0)))
    kx[0] = 1e-6
    ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX**2 + KY**2
    P, LA, UA = lu(matA)
    
    start_time = time.time()
    solution_fft = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, fft_stream, matA, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y
    end_time = time.time()
    fft_time = end_time - start_time
    print(f"fft_time: {fft_time:.4f}")

    matA[0, 0] = 2 / (20 / 64) **2
    matA_sparse = sparse.csr_matrix(matA)
    P, LA, UA = lu(matA)

    start_time = time.time()
    solution_direct = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, direct_stream, matA, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y
    end_time = time.time()
    direct_time = end_time - start_time
    print(f"direct_time: {direct_time:.4f}")

    start_time = time.time()
    solution_lu = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, lu_stream, matA, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y
    end_time = time.time()
    lu_time = end_time - start_time
    print(f"lu_time: {lu_time:.4f}")

    start_time = time.time()
    solution_bicgstab = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, bicgstab_stream, matA_sparse, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y
    end_time = time.time()
    bicgstab_time = end_time - start_time
    print(f"bicgstab_time: {bicgstab_time:.4f}")

    start_time = time.time()
    solution_gmres = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, gmres_stream, matA_sparse, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y
    end_time = time.time()
    gmres_time = end_time - start_time
    print(f"gmres[-1, -1]: {solution_gmres[-1, -1]}")
    print(f"gmres_time: {gmres_time:.4f}")

    # Try out different initial conditions
    # Two oppositely charged Gaussian vortices
    w_opposite = (5 * np.exp(-X**2 - (Y + 2)**2) - 5 * np.exp(-X**2 - (Y - 2)**2)).flatten()
    
    # Two same-charge Gaussian vortices
    w_samecharged = (5 * np.exp(-X**2 - (Y + 2)**2) + 5 * np.exp(-X**2 - (Y - 2)**2)).flatten()
    
    # Two pairs of opposite-charge vortices colliding 
    w_collide = (3.8 * np.exp(-((X - 1) ** 2) - ((Y + 1.4) ** 2))  
        -3.6 * np.exp(-((X - 1) ** 2) - ((Y - 1.4) ** 2))  
        +4.2 * np.exp(-((X + 1) ** 2) - ((Y + 1.1) ** 2))  
        -3.9 * np.exp(-((X + 1) ** 2) - ((Y - 1.1) ** 2))).flatten()
    
    # random assortment of vortices
    w_random = (
        0.8 * np.exp(-((X - 3) ** 2) - ((Y + 4) ** 2))  
        -0.6 * np.exp(-((X + 5) ** 2) / 2 - ((Y - 3) ** 2) / 1.5)  
        +1.2 * np.exp(-((X + 1) ** 2) / 1.2 - ((Y + 1) ** 2) / 0.8)  
        -0.9 * np.exp(-((X - 4) ** 2) / 0.9 - ((Y - 4) ** 2) / 1.1)  
        +0.7 * np.exp(-((X + 6) ** 2) / 1.8 - ((Y - 6) ** 2) / 1.4)  
        -0.5 * np.exp(-((X - 7) ** 2) / 1.6 - ((Y + 7) ** 2) / 1.3)  
        +1.0 * np.exp(-((X - 2) ** 2) / 1.3 - ((Y + 3) ** 2) / 0.9)  
        -1.3 * np.exp(-((X + 4) ** 2) / 1.1 - ((Y - 5) ** 2) / 1.6)  
        +0.9 * np.exp(-((X - 6) ** 2) / 1.4 - ((Y + 2) ** 2) / 1.2)  
        -0.8 * np.exp(-((X + 3) ** 2) / 1.5 - ((Y + 6) ** 2) / 1.8)  
        +0.6 * np.exp(-((X - 5) ** 2) / 1.7 - ((Y - 1) ** 2) / 1.2)  
        -1.1 * np.exp(-((X + 7) ** 2) / 1.9 - ((Y - 7) ** 2) / 1.4)  
    ).flatten()

    w0 = w_random
    solution = solve_ivp(vorticity_rhs, tp, w0, args = (nu, n, N, fft_stream, matA, matB, matC, KX, KY, K, LA, UA, P), t_eval=tspan, method='RK45').y

    # plotting -------------------------------------------------------------
    sol_to_plot = solution_gmres
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(sol_to_plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet', origin='lower')
    fig.colorbar(cax, ax=ax, label='Vorticity')
    ax.set_title('Vorticity Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    def update(frame):
        ax.set_title(f'Vorticity Field at t = {frame * 0.5:.2f}')
        cax.set_data(sol_to_plot[:, frame].reshape((n, n)))
        return cax,
    anim = FuncAnimation(fig, update, frames=sol_to_plot.shape[1], blit=True)
    anim.save('./HW5/vorticity_evolution_gmres.gif', writer='imagemagick', fps=2)

if __name__ == '__main__':
    main()



