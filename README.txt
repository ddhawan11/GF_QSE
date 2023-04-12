experiment_QSE: a simple program (that we can read, debug, and refactor together) to perform 
                'quantum subspace expansion' calculations (https://arxiv.org/abs/1603.05681) 
                and extract correlation functions in real/imaginary-time and/or in frequency

Workflow: Hamiltonian using generate_hamiltonian -> VQE Ground state from ground_state_from_vqe -> GF through QSE or FCI from excited_states_from_qse -> Energy calculation using LW functional from energy_from_GF.

energy_from_GF: Python code to calculate self-energy and energy(using LW functional) from the GF obtained either through fci_green.py or main_freq.py, both present in excited_states_from_qse

results: Results for H4 chain and ring, contains all the self-energy and Green's function files as well.

Extras: Error Propagation Presentation(Error_propagation.pptx) and Final presentation(Final.pptx)
