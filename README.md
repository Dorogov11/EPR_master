This code contains convenient machinery for calculating some design parameters (such as coupling g) from pyEPR analysis and their formatted output.
© 2026 Aleksandr Dorogov

Class needs to be aware of modes corresponding to qubits. There are multiple options to provide it. In the order of priority (e.g. if 1) and 2) are given, 1) will be used):
        1) You can provide the indices of qubit modes as numpy.array() to qubits_indices argument.
        2) You can specify the names of the modes and provide them via 'modes_names' argument. The class will automatically interpret modes which contain 'Qubit' or 'qub' in their names as qubits. Names of non-qubit modes are just for reference.
        3) You can provide the information on the number of different modes in your setup via arguments num_of_qubs, num_of_cavs, num_of_rrs. The modes will then be recognised automatically.
        If nothing was provided, option 3) is used with default values num_of_qubs=1, num_of_cavs=1, num_of_rrs=1.
        For automatic mode recognition the following algorithm is used:
        num_of_qubs modes with highest anharmonicities are interpreted to be qubits. Out of other modes, num_of_cavs modes are interpreted to be cavities. Then, num_of_rrs modes with the lowest frequencies that were not assigned yet are interpreted as RRs. Other modes' (if any) names are 'Mode {number of mode}'. Therefore, num_of_qubs + num_of_cavs + num_of_rrs can be less than the number of modes but can not be greater.