from pypower.api import runpf as prunpf

def runpf(mpc, ppopt, fname):
    return prunpf(mpc, ppopt, fname=fname)
