import casadi as cs
import pkg_resources

casadi_version = pkg_resources.get_distribution('casadi').version
print("CASADI MAJOR: ", casadi_version[0])
print("CASADI MINOR: ", casadi_version[2])

sym_t = cs.MX

x = sym_t.sym('x', 3)
u = sym_t.sym('u', 2)

f = cs.sumsqr(x) + cs.sumsqr(u)

print ("f: ", f)

Ff = cs.Function('my_func', {'x': x, 'u': u, 'f': f}, ['x', 'u'], ['f'])

print ("Ff: ", Ff)

Jac_Ff = Ff.jac()

print ("Jac_Ff: ", Jac_Ff)

print("Jac_Ff_DfDx: ", Jac_Ff(x=[1,1,1], u=[1,1])["DfDx"])
print("Jac_Ff_DfDu: ", Jac_Ff(x=[1,1,1], u=[1,1])["DfDu"])

Jfx = cs.jacobian(f,x)
FJfx = cs.Function('my_func2', {'x': x, 'u': u, 'Jfx': Jfx}, ['x', 'u'], ['Jfx'])
print("Jfx: ", FJfx(x=[1,1,1], u=[1,1]))

Jfu = cs.jacobian(f,u)
FJfu = cs.Function('my_func3', {'x': x, 'u': u, 'Jfu': Jfu}, ['x', 'u'], ['Jfu'])
print("Jfu: ", FJfu(x=[1,1,1], u=[1,1]))

