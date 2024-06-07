### Parallel solver for the Laplace equation

This code can be used to solve the Laplace equation in parallel in any square domain in $\mathbb{R}^2$, with Dirichlet boundary conditions. The boundary conditions,
forcing term, domain extrema, number of points to use in the mesh, tolerance for the error and maximum number of iterations are defined in the data.json file in the main directory.
The equation is solved numerically in parallel through MPI instructions, and any arbitrary number of processes can be used, not necessarily a multiple of the number of rows in the mesh.
The code returns the time spent for the solution of the equation, which can be used for benchmarking. 
