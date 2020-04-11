from horizon import *

def final_time(T, T_min):
    g = [sum(T)]
    g_min = [T_min]
    g_max = [1e3]

    return g, g_min, g_max

