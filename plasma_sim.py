import numpy as np
import matplotlib.pyplot as plt

L = 4 * np.pi           # Domain length (periodic)
Ng = 128                # Number of grid points
N = 100000              # Total number of particles
dt = 0.1                # Time step 
v_drift = 1.0           # Drift velocity of the two beams
n_steps = 801           # Number of simulation steps

dx = L / Ng             # Cell size
x_grid = np.linspace(0, L, Ng, endpoint=False)

# Particle Initialization: Half of the particles move to the right, half to the left
pos_beam1 = np.linspace(0, L, N//2, endpoint=False)
pos_beam2 = np.linspace(0, L, N//2, endpoint=False)

pos = np.concatenate([pos_beam1, pos_beam2])

# Setting velocities for both beams
vel = np.ones(N) * v_drift
vel[:N//2] = v_drift
vel[N//2:] = -v_drift

# Perturbation (for the instability)
k_perturb = 2 * np.pi / L
pos += 0.1 * np.sin(k_perturb * pos)
pos %= L

history_ke = []    # Kinetic Energy
history_fe = []    # Field Energy
history_time = []  

def density(pos, N, Ng, dx):
    
    # Weighting: Calculates charge density on the grid from particle positions.
    
    rho = np.zeros(Ng)
    indices = np.floor(pos / dx).astype(int)
    dist = (pos / dx) - indices

    np.add.at(rho, indices % Ng, 1.0 - dist)
    np.add.at(rho, (indices + 1) % Ng, dist)
  
    return (rho * Ng / N) - 1.0 

def field(rho, Ng, dx):
    
    # Field Solver: Solves Poisson's equation using FFT.
    
    rho_k = np.fft.fft(rho)
    k = np.fft.fftfreq(Ng, d=dx) * 2 * np.pi
  
    # Avoid division by zero at k=0 mode
    k[0] = 1.0           
  
    E_k = 1j * rho_k / k
    E_k[0] = 0 
    E_k[Ng//2] = 0 
    
    # Transform back to real space
    E = np.real(np.fft.ifft(E_k))
    
    return E

# Initialization of the Leapfrog method
rho_0 = density(pos, N, Ng, dx)
E_0 = field(rho_0, Ng, dx)

idx = np.floor(pos / dx).astype(int)
fracs = (pos / dx) - idx
E_p = E_0[idx % Ng] * (1.0 - fracs) + E_0[(idx + 1) % Ng] * fracs

# Half-step backward velocity kick
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
    E = field(rho, Ng, dx)

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

    # Plot Phase Space (x, v) visualization
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
