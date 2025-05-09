# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time  

# Analytical solution as per the problems
def analytical(Gamma, u, rho, L, n):
    
    x = np.linspace(0, L, n)
    Pe = rho * u * L / Gamma

    if abs(Pe) < 1e-8:
        phi = x / L
    else:
        exp_term = np.exp(Pe * x / L)
        phi = (exp_term - 1.0) / (np.exp(Pe) - 1.0)

    return x, phi

# TDMA-based solver using 3 different schemes
def solve_conv_diff(scheme, Gamma, rho, u, L, n):
    """
    Solve 1D convection–diffusion with specified scheme using TDMA.
    scheme: 'cds', 'upwind', or 'hybrid'
    Gamma: diffusion coefficient
    u: velocity
    rho: density
    L: domain length
    n: number of grid points
    
    Returns:
    x: grid points
    phi: solution
    execution_time: time taken to solve in seconds
    """
    # Start timing
    start_time = time.time()
    
    dx = L / (n - 1)
    x = np.linspace(0, L, n)

    # Diffusive and convective coefficients
    D = Gamma / dx
    F = rho * u

    # Coefficient arrays
    aW = np.zeros(n)
    aE = np.zeros(n)
    aP = np.zeros(n)
    d  = np.zeros(n)

    scheme = scheme.lower()
    if scheme == 'cds':
        for i in range(1, n-1):
            aW[i] = D + F/2
            aE[i] = D - F/2
            aP[i] = aW[i] + aE[i]

    elif scheme == 'upwind':
        for i in range(1, n-1):
            aW[i] = D + F
            aE[i] = D
            aP[i] = aW[i] + aE[i]

    elif scheme == 'hybrid':
        for i in range(1, n-1):
            # This is taken from Numerical Heat tranfer and fluid flow book
            aE[i] = max(0, D - F/2, -F)
            aW[i] = max(0, D + F/2,  F)
            aP[i] = aW[i] + aE[i]

    else:
        raise ValueError("Scheme must be 'cds', 'upwind', or 'hybrid'.")

    aE[n-1] = 0
    aW[0] = 0
    
    # Boundary conditions
    phi = np.zeros(n)
    phi[0], phi[-1] = 0.0, 1.0

    # TDMA begins 
    P = np.zeros(n)
    Q = np.zeros(n)

    # Forward sweep
    for i in range(1, n-1):
        if i == 1:
            P[i] = aE[i] / aP[i]
            Q[i] = (d[i] + aW[i] * phi[0]) / aP[i]
        else:
            denom = aP[i] - aW[i] * P[i-1]
            P[i] = aE[i] / denom
            Q[i] = (d[i] + aW[i] * Q[i-1]) / denom

    # Back substitution
    for i in range(n-2, 0, -1):
        phi[i] = P[i] * phi[i+1] + Q[i]
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return x, phi, execution_time

# Function for calling the solver and Post processing 
def main():
    # Problem parameters
    rho = 1.0      # Density
    L = 1.0        # Length of domain
    u = 3.0        # Velocity
    Gamma = 0.01   
    
    # Array consisting of different grid sizes for grid convergence study and computation time comparision
    grid_sizes = [11, 26, 51, 101, 251, 501, 1001, 5001]
    
    for n in grid_sizes:
        print(f"\nGrid size: {n} points")
        print("-" * 50)
        
        schemes = ['cds', 'upwind', 'hybrid']
        results = {}
        execution_times = {}

        # Solve using each scheme and measure execution time
        for sch in schemes:
            x_num, phi_num, exec_time = solve_conv_diff(sch, Gamma, rho, u, L, n)
            results[sch] = (x_num, phi_num)
            execution_times[sch] = exec_time
            print(f"{sch.upper()} scheme execution time: {exec_time:.6f} seconds")

        # Get analytical solution
        x_anal, phi_anal = analytical(Gamma, u, rho, L, n)
        
        # Plotting solution comparison
        plt.figure(figsize=(10, 6))
        for sch in schemes:
            x_num, phi_num = results[sch]
            plt.plot(x_num, phi_num, label=f"{sch} ({execution_times[sch]:.6f}s)")
        plt.plot(x_anal, phi_anal, '--', label="analytical")
        plt.xlabel('x')
        plt.ylabel(r'$\phi$')
        plt.title(f'Comparison of Schemes (Grid Size: {n} points)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'scheme_comparison_n{n}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Compare execution times across different grid sizes
    compare_grid_sizes(grid_sizes, Gamma, rho, u, L)

def compare_grid_sizes(grid_sizes, Gamma, rho, u, L):
    """Compare execution times across different grid sizes for all schemes"""
    schemes = ['cds', 'upwind', 'hybrid']
    all_times = {scheme: [] for scheme in schemes}
    
    for n in grid_sizes:
        for scheme in schemes:  # Numerical schemes only
            _, _, exec_time = solve_conv_diff(scheme, Gamma, rho, u, L, n)
            all_times[scheme].append(exec_time)
    
    # Plot execution time vs grid size
    plt.figure(figsize=(10, 6))
    
    for scheme in schemes:
        plt.plot(grid_sizes, all_times[scheme], 'o-', label=scheme)
    
    plt.xlabel('Grid Size (number of points)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Grid Size')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('execution_time_vs_grid_size.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a table of execution times
    print("\nExecution Time Summary (seconds)")
    print("-" * 60)
    header = "Grid Size | " + " | ".join(f"{scheme:10s}" for scheme in schemes)
    print(header)
    print("-" * 60)
    
    for i, n in enumerate(grid_sizes):
        row = f"{n:9d} | " + " | ".join(f"{all_times[scheme][i]:10.6f}" for scheme in schemes)
        print(row)

if __name__ == '__main__':
    main()