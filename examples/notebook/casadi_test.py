import casadi
import numpy as np

x = casadi.MX.sym('x', 3)
u = casadi.MX.sym('p', 2)
f = x*x + x + casadi.vertcat(u*u + u, 0)

F = casadi.Function('f', [x, u], [f], ['x', 'u'], ['f'])
dF = F.jac()
print(dF(x=[0, 0, 0], u=[0, 0]))
dF = F.jacobian().jacobian()
print(dF(x=[1, 1, 1], u=[1, 1]))

xu = casadi.vertcat(x, u)

Q1 = np.arange(25).reshape((5,5))/2
Q2 = 10*np.arange(25).reshape((5,5))/2
Q3 = 0.1*np.arange(25).reshape((5,5))/2
AB = np.arange(3*5).reshape((3,5))

f = casadi.vertcat( casadi.dot(xu, Q1@xu),
                    casadi.dot(xu, Q2@xu),
                    casadi.dot(xu, Q3@xu)
                    ) + AB@xu
F = casadi.Function('f', [x, u], [f], ['x', 'u'], ['f'])
dFdX = F.jacobian('f', 'x')
dFdU = F.jacobian('f', 'u')
# ddF = dF.jacobian()

# H = ddF(x=[0, 0, 0], u=[0, 0])['jac'].toarray()[:, 0:5]