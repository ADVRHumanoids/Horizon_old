import pkg_resources
from Horizon.solvers.nilqr import jac
import casadi as cs
import numpy as np

def jacobian(dict, var_string_list, function_string):
    f = {}
    f[function_string] = dict[function_string]

    vars_dict = {}
    X = []
    for var in var_string_list:
        vars_dict[var] = dict[var]
        X.append(dict[var])

    jac_list = []
    jac_id_list = []
    for var in var_string_list:
        id = "D"+function_string+'D'+var
        jac_id_list.append(id)
        jac_list.append(cs.jacobian(f[function_string], vars_dict[var]))


    return cs.Function('jacobian', X, jac_list, var_string_list, jac_id_list)


def hessian(dict, var_string_list, function_string):
    f = {}
    f[function_string] = dict[function_string]

    vars_dict = {}
    X = []
    for var in var_string_list:
        vars_dict[var] = dict[var]
        X.append(dict[var])

    hessian_list = []
    hessian_id_list = []
    for var in var_string_list:
        #id = "D" + function_string + 'D' + var
        jacobian =  cs.jacobian(f[function_string], vars_dict[var])
        for var2 in var_string_list:
            id = "DD" + function_string + 'D' + var + 'D' + var2
            hessian_id_list.append(id)
            hessian_list.append(cs.jacobian(jacobian, vars_dict[var2]))

    return cs.Function('hessian', X, hessian_list, var_string_list, hessian_id_list)




sym_t = cs.MX

x = sym_t.sym('x', 1)
u = sym_t.sym('u', 1)

#f = 2*x*x + u*u + ux
f = 2.*cs.sumsqr(x) + cs.sumsqr(u) + u*x

print ("f: ",f)

Ff = cs.Function('my_func', {'x': x, 'u': u, 'f': f}, ['x', 'u'], ['f'])

print ("Ff: ", Ff)
print (Ff)

Jac_Ff = Ff.jac()

#print("Jac_Ff", Jac_Ff)

#Jacobian:
#DfDx = 4x + u
#DfDu = 2u + x
print("Jac_Ff_DfDx: ", Jac_Ff(x=[1], u=[1])["DfDx"])
print("Jac_Ff_DfDu: ", Jac_Ff(x=[1], u=[1])["DfDu"])

print("")
##SUGGESTED ALTERNATIVE
df_dx = cs.jacobian(f, x)
df_du = cs.jacobian(f, u)

my_jac = cs.Function('jacfcn', [x, u], [df_dx, df_du], ['x', 'u'], ['DfDx', 'DfDu'])

print("Jac_Ff_DfDx: ", my_jac(x=[1], u=[1])["DfDx"])
print("Jac_Ff_DfDu: ", my_jac(x=[1], u=[1])["DfDu"])

print("")
## OUR FUNC
my_jac2 = jacobian({'x': x, 'u': u, 'f': f}, ['x', 'u'], 'f')
print("Jac_Ff_DfDx: ", my_jac2(x=[1], u=[1])["DfDx"])
print("Jac_Ff_DfDu: ", my_jac2(x=[1], u=[1])["DfDu"])

print("")
###OUR JAC
jacF, jack = jac({'x': x, 'u': u, 'f': f}, ['x', 'u'], 'f')
print("Jac_Ff_DfDx: ", jacF(x=[1], u=[1])["DfDx"])
print("Jac_Ff_DfDu: ", jacF(x=[1], u=[1])["DfDu"])
print("jac: ", jack)

print("")
#Hessian
#DDfDxDx = 4
#DDfDuDu = 2
#DDfDxDu = 1
#DDfDuDx = 1
Hes_Ff = Jac_Ff.jac()
#print("Hes_Ff", Hes_Ff)
print("Hes_Ff_DDfDxDx: ", Hes_Ff(x=[1], u=[1])["DDfDxDx"])
print("Hes_Ff_DDfDuDu: ", Hes_Ff(x=[1], u=[1])["DDfDuDu"])
print("Hes_Ff_DDfDxDu: ", Hes_Ff(x=[1], u=[1])["DDfDxDu"])
print("Hes_Ff_DDfDuDx: ", Hes_Ff(x=[1], u=[1])["DDfDuDx"])

print("")
##SUGGESTED ALTERNATIVE
ddf_dxdx = cs.jacobian(df_dx, x)
ddf_dudu = cs.jacobian(df_du, u)
ddf_dxdu = cs.jacobian(df_dx, u)
ddf_dudx = cs.jacobian(df_du, x)

my_ass = cs.Function('hesfcn', [x, u], [ddf_dxdx, ddf_dudu, ddf_dxdu, ddf_dudx], ['x', 'u'], ['DDfDxDx', 'DDfDuDu', 'DDfDxDu', 'DDfDuDx'])
print("")
print("Hes_Ff_DDfDxDx: ", my_ass(x=[1], u=[1])["DDfDxDx"])
print("Hes_Ff_DDfDuDu: ", my_ass(x=[1], u=[1])["DDfDuDu"])
print("Hes_Ff_DDfDxDu: ", my_ass(x=[1], u=[1])["DDfDxDu"])
print("Hes_Ff_DDfDuDx: ", my_ass(x=[1], u=[1])["DDfDuDx"])

print("")
## OUR FUNC
my_ass2 = hessian({'x': x, 'u': u, 'f': f}, ['x', 'u'], 'f')
print("Hes_Ff_DDfDxDx: ", my_ass2(x=[1], u=[1])["DDfDxDx"])
print("Hes_Ff_DDfDuDu: ", my_ass2(x=[1], u=[1])["DDfDuDu"])
print("Hes_Ff_DDfDxDu: ", my_ass2(x=[1], u=[1])["DDfDxDu"])
print("Hes_Ff_DDfDuDx: ", my_ass2(x=[1], u=[1])["DDfDuDx"])

print("")
###OUR HESSIAN
d = dict(list({'x': x, 'u': u}.items()) + list(jack.items()))
print("d: ", d)
print("jac.keys(): ", jack.keys())
hesF, hess = jac(d, ['x', 'u'], jack.keys())
print("Hes_Ff_DDfDxDx: ", hesF(x=[1], u=[1])["DDfDxDx"])
print("Hes_Ff_DDfDuDu: ", hesF(x=[1], u=[1])["DDfDuDu"])
print("Hes_Ff_DDfDxDu: ", hesF(x=[1], u=[1])["DDfDxDu"])
print("Hes_Ff_DDfDuDx: ", hesF(x=[1], u=[1])["DDfDuDx"])

print("hess: ", hess)
print("")
### Next derivative
for key in jack.keys(): del d[key]
print("d: ", d)
d = dict(list(d.items()) + list(hess.items()))
pippoF, pippo = jac(d, ['x', 'u'], hess.keys())
print("pippo: ", pippo)


sym_t = cs.MX

X = sym_t.sym('x', 1)

#f = 2*x*x + u*u + ux
f = X*X*X*X
f_map = {'f': f}
var_map = {'x': X}
var_string_list = ['x']
for i in range(10):
    F, f = jac(dict(list(var_map.items()) + list(f_map.items())), var_string_list, f_map.keys())
    print("f: ", f)
    f_map = f

f = X*X*X*X
f_map = {'f': f}
var_map = {'x': X}
for i in range(10):
    F, f = jac(dict(list(var_map.items()) + list(f_map.items())), var_string_list, f_map.keys())
    print("f: ", f)
    h = F(x=X)["DfDx"]
    f_map = {'f': h}
