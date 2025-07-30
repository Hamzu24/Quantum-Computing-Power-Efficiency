import qiskit
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import os
n_qubits = 6
device_name = "noisy_sim"
shots = 1000
print(os.getcwd())  # Gets current working directory
from abc.xyz import CircuitSubmitter
from _helpers.circuit_submitter import CircuitSubmitter
submitter = CircuitSubmitter("qft", device_name)

def qft(n_qubits, inverse=False):
    qc = QuantumCircuit(n_qubits, n_qubits)

    for i in (range(n_qubits)):
        qc.h(i)
        for j in (range(i + 1, n_qubits)):
            lam = np.pi * (2.0 ** (i - j))
            qc.cp(lam, j, i)

    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - i - 1)

    if inverse:
        qc = qc.inverse()
        
    return qc


qc = QuantumCircuit(n_qubits, n_qubits)

# Random initial state. Seed for reproducibility
np.random.seed(0)
random_initialisation = np.random.randint(0, 2, n_qubits)

for i, flag in enumerate(random_initialisation):
    if flag == 1:
        qc.x(i)
qc = qc & qft(n_qubits)

for i in range(n_qubits):
    qc.rz(np.pi / 2 ** i, i)

qc = qc & qft(n_qubits, inverse=True)
qc.measure_all()

qc.draw(style="clifford", output="mpl")

submitter.submit_circuits(shots=shots, verbatim=False, skip_asking=True, qasm_strs=[qc.qasm()])
counts = submitter.retrieve_counts()
print(counts)

# Convert base 2 into base 10
counts = {int(key, 2): value for key, value in counts[0].items()}
# Sort according to ascending order of bit strings
counts = np.array([counts[i] if i in counts.keys() else 0 for i in range(2 ** n_qubits)])
# Convert to probabilities
hardware_probs = counts / shots
print(hardware_probs)

exact_probs = np.zeros(len(hardware_probs))
exact_probs[(sum(v * (2 ** i) for i, v in enumerate(reversed(random_initialisation))) + 1) % (2 ** n_qubits)] = 1

plt.figure(figsize=(16, 5))
plt.plot(exact_probs, label="Exact probabilities")
plt.plot(hardware_probs, label="Hardware results")
plt.xlabel("Bitstrings (converted to decimal numbers)")
plt.ylabel("Probability")
plt.legend()
plt.savefig(submitter.benchmark_path + "/plot.png", format="png")
plt.show()

def fidelity(P1, P2):
    assert len(P1) == len(P2)
    
    f = 0
    for i in range(len(P1)):
        f += np.sqrt(P1[i] * P2[i])
    
    f = f ** 2
    
    return f

uniform_dist = np.array([1 for _ in exact_probs]) / len(exact_probs)

f_ideal_and_output = fidelity(exact_probs, hardware_probs)
f_ideal_and_uniform = fidelity(exact_probs, uniform_dist)


f_ideal_and_hardware = fidelity(exact_probs, hardware_probs)

normalised_fidelity = max((f_ideal_and_hardware - f_ideal_and_uniform) / (1 - f_ideal_and_uniform), 0)
print(f"The normalised fidelity between the exact result and the hardware result is {normalised_fidelity}")

# Save into data directory
result_str = (f"Quantum Fourier Transform benchmark with n_qubits = {n_qubits}, random initialization {random_initialisation}\n"
                f"Exact (ideal) probabilities: {exact_probs}\n" 
                f"Obtained (hardware) probabilities: {hardware_probs}\n" 
                f"The normalised fidelity between the exact result and the hardware result is {normalised_fidelity}")
with open(submitter.benchmark_path + "/results.txt", "w+") as f:
    f.write(result_str)
print(result_str)
print(f"The above results are saved into {submitter.benchmark_path + '/results.txt'}")

(f_ideal_and_hardware - f_ideal_and_uniform) / (1 - f_ideal_and_uniform)
