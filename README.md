# Gravitational_Waves
This block I worked on simulating and visualizing the gravitational wave signal from a binary star system. First, I used work done by Hugo Wahlquist to animate the gravitational wave signal as a function of the orbital angle. This was done for an arbitrarily oriented binary star system, and could accept any numerically feasible star masses, orbital eccentricity, and orbital period. Next, I numerically solved Kepler’s ψ-function to relate the orbital angle to time, which I then used to determine the time dependent gravitational wave signal. In this way, I was able to generate a realistic signal that we would observe on Earth (well, probably in space for this sort of system...) due to a distant binary system; however, this solution assumed a static system. In reality, gravitational waves carry energy, and so the binary system loses energy over time. Using work done by P.C. Peters, I was able to account for the energy lost in the binary system over a relatively short time period. I failed to achieve longer integrations as I neverg got an adaptive step size algorithm working.

* Gravity_Waves_Final_Paper.pdf: write up of my methods and findings.
* Presentation1.pptx: presentation I gave at the end of the block.
* \Orbital_Dynamics: Main code directory
  * \Keppler_Equation: solving Keppler's equation using Newton's method
  * Binary_Star_Waves.py: main code file which defines the Binary_System class which has functions for calculating and visualizing the gravitational wave signal from a binary star system along with the binary orbit.
