using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters t x
@variables u(..)

Dtt = Differential(t)^2
Dxx = Differential(x)^2
Dt = Differential(t)

xmin = 0.0
xmax = 1.0

domains = [x ∈ Interval(xmin , xmax),
           t ∈ Interval(0.0 , 1.0)]

c = 1
eq = Dtt(u(t,x)) ~ c^2 * Dxx(u(t,x))

bcs = [u(0,x) ~ x*(1-x),
       Dt(u(0,x)) ~ 0,
       u(0,t) ~ 0,
       u(1,t) ~ 0 ]

#Building the NN
chain = Lux.Chain(Lux.Dense(2,30 , Lux.σ), Lux.Dense(30, 16, Lux.σ), Lux.Dense(16, 1))
#Defining the PDE system
@named pde_system = PDESystem(eq, bcs, domains, [t,x], [u(t, x)])

#Discretization 
dx = 0.1
discretization = PhysicsInformedNN(chain,GridTraining(dx)) #selecting GridTraining as our training strategy.
prob = discretize(pde_system, discretization)

#Optimization
opt = OptimizationOptimJL.BFGS()

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

result = Optimization.solve(prob, opt, callback = callback, maxiters = 1000)
phi = discretization.phi      #This variable phi contains the values of the dependent variable (i.e., the solution to the PDE system) at the specified grid points.

using Plots

ts, xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains] # ts and xs are arrays containing values at which PINN and analytic soln are evaluated.
xs 
ts #xs and ts are values from 0.0 to 1.0 with steps of 0.1

function analytic_sol_func(t, x)
    sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(c * k * pi * t) for k in 1:2:50000])
end

u_predict = reshape([first(phi([t, x], result.u)) for t in ts for x in xs],
    (length(ts), length(xs)))
u_real = reshape([analytic_sol_func(t, x) for t in ts for x in xs],
    (length(ts), length(xs)))

diff_u = abs.(u_predict .- u_real)
p1 = plot(ts, xs, u_real, linetype = :contourf, title = "analytic");
p2 = plot(ts, xs, u_predict, linetype = :contourf, title = "predict");
p3 = plot(ts, xs, diff_u, linetype = :contourf, title = "error");
plot(p1, p2, p3)


