import numpy as np
import matplotlib.pyplot as plt

L = 4 * np.pi           # Domain length (periodic)
Ng = 128                # Number of grid points
N = 400000              # Total number of particles
dt = 0.1                # Time step 
v_drift = 1.0           # Drift velocity of the two beams
n_steps = 801           # Number of simulation steps

dx = L / Ng             # Cell size
x_grid = np.linspace(0, L, Ng, endpoint=False)

# Particle initialization
pos = np.random.uniform(0, L, N)

# Velocities (two beams)
vel = np.zeros(N)
v_thermal = 0.2

vel[:N//2] = v_drift + (np.random.randn(N//2) * v_thermal)
vel[N//2:] = -v_drift + (np.random.randn(N//2) * v_thermal)

# Perturbation 
k_perturb = 2 * np.pi / L

pos += 0.1 * np.sin(k_perturb * pos)
pos %= L

history_ke = []    # Kinetic Energy
history_fe = []    # Field Energy
history_time = []  # Time steps

def density(pos, N, Ng, dx):
    
    # Calculates charge density on the grid from particle positions
    
    rho = np.zeros(Ng)
    indices = np.floor(pos / dx).astype(int)
    dist = (pos / dx) - indices

    np.add.at(rho, indices % Ng, 1.0 - dist)
    np.add.at(rho, (indices + 1) % Ng, dist)
  
    n= (rho * Ng / N)
    dens = -1.0 * (n-1.0)
    
    return dens

def field(rho, Ng, dx):
    
    # Field Solver: Solves Poisson's equation using FFT.
    
    rho_k = np.fft.fft(rho)
    k = np.fft.fftfreq(Ng, d=dx) * 2 * np.pi

    # Avoid division by zero at k=0 mode
    k[0] = 1.0           

    phi_k = rho_k / (k**2)
    phi_k[0] = 0 # Il potenziale medio è arbitrario (poniamolo a 0)
    phi = np.real(np.fft.ifft(phi_k))

    k_fixed = np.copy(k)
    k_fixed[0] = 1.0
    E_k = -1j * rho_k / k_fixed
    E_k[0] = 0 
    E_k[Ng//2] = 0 
    
    # Transform back to real space
    E = np.real(np.fft.ifft(E_k))
    phi = np.real(np.fft.ifft(phi_k))
    
    return E, phi

# Initialization of the Leapfrog method
rho_0 = density(pos, N, Ng, dx)
E_0, phi_0= field(rho_0, Ng, dx)

idx = np.floor(pos / dx).astype(int)
fracs = (pos / dx) - idx
E_p = E_0[idx % Ng] * (1.0 - fracs) + E_0[(idx + 1) % Ng] * fracs

# Half step backward velocity kick
vel += E_p * (0.5 * dt) 

def push_particles(pos, vel, E, dx, dt, L):
    
    # Push: updates velocities and positions using the Lorentz force.
    
    idx = np.floor(pos / dx).astype(int)
    frac = (pos / dx) - idx
    E_part = E[idx % Ng] * (1.0 - frac) + E[(idx + 1) % Ng] * frac
    
    # Velocity and position update
    vel -= E_part * dt
    pos += vel * dt
    
    # Periodic boundary conditions
    pos %= L
    return pos, vel

# MAIN LOOP
for step in range(n_steps):
    
    rho = density(pos, N, Ng, dx)
    E, phi = field(rho, Ng, dx)

    v_old = np.copy(vel)
    pos, vel = push_particles(pos, vel, E, dx, dt, L)
    
    # Synchronize velocity for energy calculation
    v_sync = (v_old + vel) / 2.0
    
    # Energy
    fe = 0.5 * np.sum(E**2) * dx 
    ke = 0.5 * (L / N) * np.sum(v_sync**2)
    
    history_fe.append(fe)
    history_ke.append(ke)
    history_time.append(step * dt)

    # Plot Phase Space (x, v)
    if step % 50 == 0:
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        
        skip = 10 
        plt.scatter(pos[:N//2:skip], vel[:N//2:skip], s=5, color='royalblue', alpha=0.6, label='Right Beam')
        plt.scatter(pos[N//2::skip], vel[N//2::skip], s=5, color='crimson', alpha=0.6, label='Left Beam')
        
        plt.title(f"Phase Space Evolution - Time: {step*dt:.2f}")
        plt.xlabel("Position x")
        plt.ylabel("Velocity v")
        plt.xlim(0, L)
        plt.ylim(-v_drift*5, v_drift*5)
        plt.legend(loc='upper right', markerscale=10)
        plt.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.pause(0.01)

# Energy visualization
plt.figure(figsize=(10, 6))

h_time = np.array(history_time)
h_ke = np.array(history_ke)
h_fe = np.array(history_fe)
h_total = h_ke + h_fe 

plt.plot(h_time, h_ke, label='Energia Cinetica ($E_{kin}$)', color='blue', lw=2)
plt.plot(h_time, h_fe, label='Energia Campo ($E_{field}$)', color='red', lw=2)
plt.plot(h_time, h_total, label='Energia Totale', color='black', linestyle='--', alpha=0.7)

plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.show()

plt.figure(figsize=(12, 5))

# Velocity Histogram
plt.subplot(1, 2, 1)
plt.hist(vel, bins=100, density=True, color='gray', alpha=0.8, label='Particle dist.')
plt.title(f"Velocity Histogram (Step {n_steps})")
plt.xlabel("Velocity (v)")
plt.ylabel("f(v)")
plt.xlim(-3, 3) # Adatta in base al tuo v_drift
plt.grid(True, alpha=0.3)

ax1 = plt.subplot(1, 2, 2)

# Density
color_rho = 'tab:green'
ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Density (rho)', color=color_rho)
ax1.plot(x_grid, rho, color=color_rho, alpha=0.6, linewidth=1, label='Density')
ax1.tick_params(axis='y', labelcolor=color_rho)

# Potential
ax2 = ax1.twinx()  # Crea un secondo asse Y condiviso
color_phi = 'black'
ax2.set_ylabel('Potential (phi)', color=color_phi)
ax2.plot(x_grid, phi, color=color_phi, linewidth=2, label='Potential')
ax2.tick_params(axis='y', labelcolor=color_phi)

plt.title("Potential (Phi) & Density (Rho)")
plt.tight_layout()
plt.show()
