import numpy as np
import functools
from   qiskit import *
from   qiskit.quantum_info.operators.pauli import Pauli, pauli_group
from   qiskit.aqua                         import QuantumInstance
from   qiskit.aqua.operators               import WeightedPauliOperator

def single_qubit_pauli(direction,i,n):
    zv = [0]*n
    xv = [0]*n
    if(direction=='z'): zv[i] = 1
    if(direction=='x'): xv[i] = 1
    if(direction=='y'): zv[i] = 1; xv[i] = 1
    zv = np.asarray(zv,dtype=np.bool)
    xv = np.asarray(xv,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(zv,xv))])

def measure_operator(H,circuit,instance):
    # given an operator H
    # and a circuit
    # and a quantum instance,
    # return <H> = m +/- s
    circuits = [H.construct_evaluation_circuit(
              wave_function               = circuit,
              statevector_mode            = instance.is_statevector,
              use_simulator_snapshot_mode = instance.is_statevector,
              circuit_name_prefix         = 'H')]

    to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
    result = instance.execute(to_be_simulated_circuits)

    res = np.zeros(2)
    mean,std = H.evaluate_with_result(
               result                      = result,
               statevector_mode            = instance.is_statevector,
               use_simulator_snapshot_mode = instance.is_statevector,
               circuit_name_prefix         = 'H')
    return np.real(mean),np.abs(std) # mean and standard deviation

