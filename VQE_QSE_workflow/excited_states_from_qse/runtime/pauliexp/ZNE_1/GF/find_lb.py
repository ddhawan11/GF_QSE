import qiskit
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import backend_overview

IBMQ.load_account()
provider = qiskit.IBMQ.get_provider('ibm-q')
backend = least_busy(provider.backends(simulator=False, n_qubits=5, operational=True))
print("Least busy backend:", backend.name())
#print("\nAll backends overview:\n")
#backend_overview()

