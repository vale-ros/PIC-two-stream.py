This repository has a Python program implementing an electrostatic Particle-In-Cell simulation. 
The simulation is used to study the Two-Stream Instability, and the code is specifically for 
the "Warm Beam" regime, offering a kinetic description of a plasma where electron beams possess 
a finite thermal spread rather than being monochromatic. 

This simulation serves as a numerical solver for the Vlasov-Poisson system, evolving the electron 
distribution function under the influence of self-consistent electric fields against a neutralizing 
ionic background.

The physics that is happening here is two beams of electrons that are moving in opposite directions. 
This realistic thermal spread introduces significant physical effects, such as phase mixing and the 
suppression of sharp phase-space features, providing a more accurate representation of laboratory or
astrophysical plasmas. The particle positions are initialized uniformly at random to simulate a true gas.

The code works with the PIC cycle. It puts charge density onto a grid using the linear weighting method, 
and the Poisson equation is solved in Fourier space using Fast Fourier Transforms (FFT).
The particle equations of motion are integrated using the symplectic Leapfrog scheme, which provides 
excellent energy conservation properties over long simulation times. A specific phase shift is applied
to the initial density perturbation to ensure the resulting phase-space vortex forms in the center of 
the simulation domain for better visualization.

Running this script, the user will observe the classic evolution of the instability, starting with the exponential 
growth of the electric field energy during the linear phase. As the instability saturates, the phase space
visualization reveals the formation of a central trapped particle vortex.
The code requires a standard Python 3 environment with NumPy and Matplotlib installed.
It is designed to run directly as a script, producing real-time animations of the phase space evolution 
and tracking the energy exchange between the kinetic and potential components.



