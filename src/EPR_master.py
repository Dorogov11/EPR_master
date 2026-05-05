import numpy as np
import pandas as pd
import pylab as plt
import collections
import warnings
import copy
import pyEPR as epr

class EPR_master:
    """Class contains convenient machinery for calculating some design parameters (such as coupling g) from pyEPR analysis and their formatted output.
    © 2026 Aleksandr Dorogov
    """
    
    def __init__(self, epra: epr.core_quantum_analysis.QuantumAnalysis, modes_indices: list = [0, 1, 2], modes_names: np.ndarray = None, qubits_indices: np.ndarray = None, num_of_qubs: int = 1, num_of_cavs: int = 1, num_of_rrs: int = 1) -> None:
        """Class needs to be aware of modes corresponding to qubits. There are multiple options to provide it. In the order of priority (e.g. if 1) and 2) are given, 1) will be used):
        1) You can provide the indices of qubit modes as numpy.array() to qubits_indices argument.
        2) You can specify the names of the modes and provide them via 'modes_names' argument. The class will automatically interpret modes which contain 'Qubit' or 'qub' in their names as qubits. Names of non-qubit modes are just for reference.
        3) You can provide the information on the number of different modes in your setup via arguments num_of_qubs, num_of_cavs, num_of_rrs. The modes will then be recognised automatically.
        If nothing was provided, option 3) is used with default values num_of_qubs=1, num_of_cavs=1, num_of_rrs=1.
        For automatic mode recognition the following algorithm is used:
        num_of_qubs modes with highest anharmonicities are interpreted to be qubits. Out of other modes, num_of_cavs modes are interpreted to be cavities. Then, num_of_rrs modes with the lowest frequencies that were not assigned yet are interpreted as RRs. Other modes' (if any) names are 'Mode {number of mode}'. Therefore, num_of_qubs + num_of_cavs + num_of_rrs can be less than the number of modes but can not be greater.

        Args:
            epra (epr.core_quantum_analysis.QuantumAnalysis): result of epr.QuantumAnalysis(eprd.data_filename).
            modes_indices (int, optional): the indices of modes in the model. Defaults to [0, 1, 2].
            modes_names (np.ndarray, optional): list of the names for corresponding modes, e.g. np.array(['Mode 1', 'Mode 2']). Defaults to None.
            qubits_indices (np.ndarray, optional): list of the indices of modes corresponding to qubits. If None, modes with the word 'Qubit'/'qubit'/'Qub'/'qub' in the name are chosen. If there are no such modes or modes_names were not provided, the qubit modes are defined automatically as num_of_qubs modes with the highest anharmonicity. Defaults to None.
            num_of_qubs (int, optional): defines the number of qubits for automatic mode recognition if modes_names and qubits_indices were not provided. Defaults to 1.
            num_of_cavs (int, optional): defines the number of cavities for automatic mode recognition if modes_names were not provided. Defaults to 1.
            num_of_rrs (int, optional): defines the number of RRs for automatic mode recognition if modes_names were not provided. Defaults to 1.
        """
        self.epra = epra
        if qubits_indices is not None:
            self.qubit_ind = qubits_indices
        self.modes_names = modes_names
        self.modes_indices = modes_indices
        self.number_of_modes = len(modes_indices)
        self.results = dict()
        self.number_of_variations = len(epra.variations)
        _variations_ = int(epra.variations[-1]) + 1
        # self.pyEPR_results = [None] * self.number_of_variations
        self.pyEPR_results = [None] * _variations_
        # self.losses_discarded = [False] * self.number_of_variations
        self.losses_discarded = [False] * _variations_
        try:
            self.number_of_varied_variables = len(epra._hfss_variables[epra.hfss_vars_diff_idx]['0'])
        except KeyError:
            self.number_of_varied_variables = 1
        self.num_of_qubs = num_of_qubs
        self.num_of_cavs = num_of_cavs
        self.num_of_rrs = num_of_rrs

    def print_fancy_results(self, variation_ind: str = '0'):
        """Prints the following Hamiltonian parameters extracted by pyEPR from the HFSS model in a nice way:
        Mode frequencies [GHz]
        g [MHz]
        T_1 [us] (if losses included)
        kappa/2pi [kHz] (if losses included)
        chi [MHz]
        Q-factor / 10^7 (if losses included)
        
        For Mode frequencies, g, T_1 and kappa/2pi three results are provided for each mode, corresponding to eigenmode frequencies computed by HFSS ('HFSS'), dressed mode frequencies based on 1st order perturbation theory on the 4th order expansion of the cosine ('Dressed') & numerical diagonalization result of dressed mode frequencies ('Numerical').
        Two chi matrices are calculated: analytical - analytic expression for the chis based on a cos trunc to 4th order, and using 1st order perturbation theory; numerical - numerically diagonalized chi matrix. Diag is anharmonicity, off diag is full cross-Kerr.

        Args:
            variation_ind (str, optional): the index of variation to analyze. Defaults to '0'.
        """
        if isinstance(variation_ind, str):
            valid_var_ind = variation_ind
        elif isinstance(variation_ind, int):
            valid_var_ind = str(variation_ind)
        else:
            raise TypeError("The variation index should be str (or int)")
        if not valid_var_ind in self.epra.variations:
            raise KeyError(f"There are only {self.number_of_variations} variations. Variation with index {valid_var_ind} doesn't exist.")
        pyEPR_result = self._retrieve_variation(valid_var_ind)
        refined_modes_names = np.array([("Mode " + str(number_of_mode+1)) for number_of_mode in self.modes_indices], dtype='<U23')
        qubit_ind = np.array([])
        if self.modes_names is None:
            qubit_ind, cavity_ind, rr_ind = self.automatic_mode_recognition(pyEPR_result=pyEPR_result)
            refined_modes_names = create_mode_names(sample=refined_modes_names, inds=qubit_ind, name='Qubit')
            refined_modes_names = create_mode_names(sample=refined_modes_names, inds=cavity_ind, name='Cavity')
            refined_modes_names = create_mode_names(sample=refined_modes_names, inds=rr_ind, name='RR')
        else:
            refined_modes_names = self.modes_names
        try:
            qubit_ind = self.qubit_ind
        except AttributeError:
            qubit_ind = np.arange(0, len(refined_modes_names))[np.array([(('Qubit' in mode_name) or ('qubit' in mode_name) or ('Qub' in mode_name) or ('qub' in mode_name)) for mode_name in refined_modes_names])]
        if self.number_of_modes != len(refined_modes_names):
            warnings.warn(f"The number of modes_names doesn't correspond to the number of modes of the HFSS model\n  modes_names: {len(refined_modes_names)}\n  HFSS modes: {self.number_of_modes}")
            if self.number_of_modes < len(refined_modes_names):
                refined_modes_names = self.modes_names[:self.number_of_modes]
                qubit_ind = qubit_ind[qubit_ind <= self.number_of_modes]
            else:
                refined_modes_names = self.modes_names + [("Mode " + str(number_of_mode+1)) for number_of_mode in range(len(self.modes_names), self.number_of_modes)]
        if len(qubit_ind) == 0:
            qubit_ind = np.array([np.argmax(np.diag(pyEPR_result['chi_ND']))])
        self.modes_names = refined_modes_names
        self.qubit_ind = qubit_ind
        columns_dict = {valid_var_ind: "HFSS", 0: "Dressed", 1: "Numerical"}
        indices_dict = dict(enumerate(refined_modes_names))
        if self.losses_discarded[int(valid_var_ind)]:
            warnings.warn("No lossy elements were introduced! Q-factors, T_1s and kappa/2pi will not be calculated")
        display(pd.concat([pyEPR_result['f_0'], pyEPR_result['f_1'], pyEPR_result['f_ND']],axis=1).div(1000).rename(columns=columns_dict, index=indices_dict).style.set_caption('Modes frequencies, GHz').background_gradient(axis=None))
        if self.num_of_qubs > 0:
            g_display_list = qubit_ind
        else:
            g_display_list = [0]
        for qubit_idx in g_display_list:
            display(self.get_g(pyEPR_result['f_0'], pyEPR_result['ZPF'], qubit_idx).to_frame().rename(columns={0: "g, MHz"}, index=self._get_g_indices_list(qubit_idx)).style.background_gradient(axis=None))
        if not self.losses_discarded[int(valid_var_ind)]:
            kappas_pd = pd.concat([pyEPR_result['f_0'].div(pyEPR_result['Qs']), pyEPR_result['f_1'].div(pyEPR_result['Qs']), pyEPR_result['f_ND'].div(pyEPR_result['Qs'])],axis=1)
            T_1_pd = 1/(2 * np.pi * kappas_pd)
            display(T_1_pd.rename(columns=columns_dict, index=indices_dict).style.set_caption('T_1, us').background_gradient(axis=None))
            display(kappas_pd.mul(1e3).rename(columns=columns_dict, index=indices_dict).style.set_caption('kappa/2pi, kHz').background_gradient(axis=None))
        display(pyEPR_result['chi_O1'].rename(columns=indices_dict, index=indices_dict).style.set_caption('chi analytical, MHz').background_gradient(axis=None))
        display(pyEPR_result['chi_ND'].rename(columns=indices_dict, index=indices_dict).style.set_caption('chi numerical, MHz').background_gradient(axis=None))
        if not self.losses_discarded[int(valid_var_ind)]:
            display(pyEPR_result['Qs'].to_frame().div(1e7).rename(columns={valid_var_ind: "Q-factor, 10^7"}, index=indices_dict).style.background_gradient(axis=None)) #dict(zip(self.modes_indices, self.modes_names))
        
    def print_variations(self, automatic_mode_recognition: bool = True):
        """Prints design parameters for all the variations from Optimetrics. It will first run the print_fancy_results() function, if it was not run before.

        Args:
            automatic_mode_recognition (bool, optional): whether to use modes set by user or define them automatically for each variation. Defaults to True.
        """
        T_1_list = []
        anharmonicity_list = []
        frequencies = []
        qub_freqs = []
        kappas = []
        Q_factors = []
        chis_qub = []
        g_list = []
        results_list = []
        columns_list = []
        try:
            qubit_ind = self.qubit_ind
        except AttributeError:
            self.print_fancy_results()
            qubit_ind = self.qubit_ind
        modes_names_without_qubits = np.delete(self.modes_names, self.qubit_ind)
        for var_index in self.epra.variations:
            var_result = self._retrieve_variation(var_index=var_index)
            if automatic_mode_recognition:
                qubit_ind = self.automatic_mode_recognition(var_result)[0]
            Q_factors_pd = var_result['Qs']
            kappas_pd = var_result['f_ND'].div(Q_factors_pd)
            T_1_pd = 1/(2 * np.pi * kappas_pd)
            T_1_list += [[T_1_pd.iloc[qub_mode_index] for qub_mode_index in qubit_ind]]
            anharmonicity_list += [[var_result['chi_ND'][qub_mode_index].iloc[qub_mode_index] for qub_mode_index in qubit_ind]]
            # chis += [[[var_result['chi_ND'][ii].iloc[jj] for ii in range(jj+1, self.number_of_modes)] for jj in range(self.number_of_modes-1)]]
            chis_qub.append(np.delete(var_result['chi_ND'][qubit_ind].to_numpy(), qubit_ind))
            local_freqs = var_result['f_ND'].div(1000).to_numpy()
            local_kappas = kappas_pd.mul(1000).to_numpy()
            local_Q_factors = Q_factors_pd.div(10000000).to_numpy()
            qub_freqs += [[local_freqs[qub_mode_index] for qub_mode_index in qubit_ind]]
            frequencies.append(np.delete(local_freqs, qubit_ind))
            kappas.append(np.delete(local_kappas, qubit_ind))
            Q_factors.append(np.delete(local_Q_factors, qubit_ind))
            g_list += [[self.get_g(var_result['f_0'], var_result['ZPF'], qub_mode) for qub_mode in qubit_ind]]
        g_list = np.array(g_list)
        if not all(self.losses_discarded):
            for ii, qub_mode_index in enumerate(self.qubit_ind):
                results_list.append(np.array(T_1_list)[:, ii])
                columns_list.append(', '.join(['T_1', self.modes_names[qub_mode_index], 'us']))
        for jj, qub_mode in enumerate(self.qubit_ind):
            qub_mode_gs = g_list[:, jj, :]
            g_inds = self._get_g_indices_list(qub_mode)
            for ii, qub_mode_g in enumerate(qub_mode_gs.T):
                results_list.append(qub_mode_g)
                columns_list.append(' '.join(['g', g_inds[ii], 'MHz']))
        for ii, qub_mode_index in enumerate(self.qubit_ind):
                results_list.append(np.array(qub_freqs)[:, ii])
                columns_list.append(', '.join(['Freq', self.modes_names[qub_mode_index], 'GHz']))
        for ii in range(len(modes_names_without_qubits)):
            results_list.append(np.array(frequencies)[:, ii])
            columns_list.append(', '.join(['Freq', modes_names_without_qubits[ii], 'GHz']))
        for ii, qub_mode_index in enumerate(self.qubit_ind):
            results_list.append(np.array(anharmonicity_list)[:, ii])
            columns_list.append(', '.join(['Anharm', self.modes_names[qub_mode_index], 'MHz']))
        for ii, qub_mode_index in enumerate(self.qubit_ind):
            for jj in range(len(modes_names_without_qubits)):
                results_list.append([chis_qub[uuu][jj] for uuu in range(self.number_of_variations)])
                columns_list.append('CHI ' + self.modes_names[qub_mode_index] + '-' + modes_names_without_qubits[jj] + ', MHz')
        # for ii in range(self.number_of_modes-1):
        #     for jj in range(self.number_of_modes - ii - 1):
        #         results_list.append([chis[uuu][ii][jj] for uuu in range(self.number_of_variations)])
        #         columns_list.append('chi ' + self.modes_names[ii] + '-' + self.modes_names[jj + ii + 1] + ', MHz')
        if not all(self.losses_discarded):
            for ii in range(len(modes_names_without_qubits)):
                results_list.append(np.array(Q_factors)[:, ii])
                columns_list.append(', '.join(['Q-factor', modes_names_without_qubits[ii], '10^7']))
                results_list.append(np.array(kappas)[:, ii])
                columns_list.append(', '.join(['kappa', modes_names_without_qubits[ii], 'kHz']))
        self.results.update(dict(zip(columns_list, results_list)))
        df = pd.DataFrame(np.column_stack(results_list), columns=columns_list)
        if self.number_of_varied_variables == 1:
            self.variation_values_list = np.array([self.epra._hfss_variables[self.epra.hfss_vars_diff_idx][var_ind].iloc[0] for var_ind in self.epra.variations])
            # self.varied_variable = self.epra._hfss_variables[self.epra.hfss_vars_diff_idx]['0'].index[0][1:]
            self.varied_variable = self.epra._hfss_variables[self.epra.hfss_vars_diff_idx].index[0][1:]
            columns_dict = dict(enumerate(self.variation_values_list))
            display(df.transpose().rename(columns=columns_dict).style.set_caption(self.varied_variable).background_gradient(axis=1))
        else:
            self.varied_variable = np.array([self.epra._hfss_variables[self.epra.hfss_vars_diff_idx]['0'].index[idx][1:] for idx in range(self.number_of_varied_variables)])
            self.variation_values_list = np.array([[self.epra._hfss_variables[self.epra.hfss_vars_diff_idx][var_ind].iloc[idx] for var_ind in self.epra.variations] for idx in range(self.number_of_varied_variables)])
            self.variation_titles_list = np.array([', '.join([': '.join([self.varied_variable[idx], self.variation_values_list[idx][int(var_ind)]]) for idx in range(self.number_of_varied_variables)]) for var_ind in self.epra.variations])
            columns_dict = dict(enumerate(self.variation_titles_list))
            display(df.transpose().rename(columns=columns_dict).style.set_caption(', '.join(self.varied_variable)).background_gradient(axis=1))
        
    def get_g(self, omegas: pd.core.series.Series, ZPFs: np.ndarray, qubit_mode: int = 1) -> pd.core.series.Series:
        """Calculates the g couplings between the qubit and other modes using zero-point fluctuations (ZPF) from pyEPR

        Args:
            omegas (pd.core.series.Series): frequencies of the modes.
            ZPFs (np.ndarray): modes' ZPFs.
            qubit_mode (int): the index of the qubit mode. Defaults to 1.

        Returns:
            pd.core.series.Series: g couplings between the qubit and other modes.
        """
        qubit_freq = omegas[qubit_mode]
        freqs = omegas.drop(qubit_mode)
        res = []
        if ZPFs.ndim > 1:
            ZPFs_ = ZPFs[:, 0]
            qubit_zpf = ZPFs_[qubit_mode]
            phis = np.delete(ZPFs_, qubit_mode)
            for ii, freq in enumerate(freqs):
                res.append(abs((freq - qubit_freq) * qubit_zpf * phis[ii] / (qubit_zpf**2 + phis[ii]**2)))
        else:
            qubit_zpf = ZPFs[qubit_mode]
            phis = np.delete(ZPFs, qubit_mode)
            for ii, freq in enumerate(freqs):
                res.append(abs((freq - qubit_freq) * qubit_zpf * phis[ii] / (qubit_zpf**2 + phis[ii]**2))[0])
        return pd.Series(res)

    def _get_g_indices_list(self, qubit_mode: int) -> dict:
        """Creates indices list for g couplings entry for the function print_fancy_results

        Args:
            qubit_mode (int): the index of the qubit mode.

        Returns:
            dict: indices list for g couplings entry for the function print_fancy_results
        """
        qubit_mode_name = self.modes_names[qubit_mode]
        refined_modes_names = np.delete(self.modes_names, qubit_mode)
        new_names = []
        for mode_name in refined_modes_names:
            new_names.append(" - ".join([qubit_mode_name, mode_name]))
        return dict(enumerate(new_names))

    def automatic_mode_recognition(self, pyEPR_result: collections.OrderedDict) -> list:
        """Recognizes modes of the model (qubit, cavity & RR)

        num_of_qubs modes with highest anharmonicities are interpreted to be qubits. Out of other modes, num_of_cavs modes are interpreted to be cavities. Then, num_of_rrs modes with the lowest frequencies that were not assigned yet are interpreted as RRs. Other modes' (if any) names are 'Mode {number of mode}'.

        Args:
            pyEPR_result (collections.OrderedDict): the result of the pyEPR.QuantumAnalysis().analyze_all_variations() or pyEPR.QuantumAnalysis().analyze_variation() function.

        Returns:
            list: [qubit_mode, cavity_mode, rr_mode]
        """
        _argsort_alpha = np.argsort(np.diag(pyEPR_result['chi_ND']))
        qubit_mode = _argsort_alpha[-self.num_of_qubs:]
        _argsort_Q = np.argsort(pyEPR_result['Qs']).to_numpy()
        for qub_mode in qubit_mode:
            _argsort_Q = np.delete(_argsort_Q, np.where(_argsort_Q==qub_mode))
        cavity_mode = _argsort_Q[-self.num_of_cavs:]
        rr_mode = np.sort(_argsort_Q[:-self.num_of_cavs])[:self.num_of_rrs]
        return [qubit_mode, cavity_mode, rr_mode]

    def plot_res(self, value: str, saveas: str = None, scale: str = 'linear'):
        """Plots results for all the variations from Optimetrics

        Args:
            value (str): the value to plot. Copy from EPR_master.print_variations() row titles.
            saveas (str, optional): name of the file to save (without extension). If None, file won't be saved.
            scale (str, optional): scale of the y-axis. Specify to 'log' if you wish to plot in logarithmic scale (usually for Q-factors). Defaults to 'linear'.
        """
        try:
            variation_values_list = self.variation_values_list
        except AttributeError:
            self.print_variations()
            variation_values_list = self.variation_values_list
        try:
            ordinate = self.results[value]
        except KeyError:
            raise KeyError(f'There is no data field \"{value}\". Check if you copied it from EPR_master.print_variations() function correctly.')
        if self.number_of_varied_variables==1:
            try:
                float_variation_values_list = np.array([float(var_value[:-2]) for var_value in variation_values_list])
                unit_name = variation_values_list[0][-2:]
            except ValueError:
                float_variation_values_list = np.array([float(var_value[:-3]) for var_value in variation_values_list])
                unit_name = variation_values_list[0][-3:]
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()
            ax.scatter(float_variation_values_list, ordinate, color='darkblue')
            ax.set(ylabel=value, xlabel=(self.varied_variable + ', ' + unit_name), yscale=scale)
            ax.minorticks_on()
            ax.grid(which='both')
        else:
            float_variation_values_list = []
            unit_name = []
            for ii in range(self.number_of_varied_variables):
                try:
                    float_variation_values_list.append([float(var_value[:-2]) for var_value in variation_values_list[ii]])
                    unit_name.append(variation_values_list[ii, 0][-2:])
                except ValueError:
                    float_variation_values_list.append([float(var_value[:-3]) for var_value in variation_values_list[ii]])
                    unit_name.append(variation_values_list[ii, 0][-3:])
            float_variation_values_list = np.array(float_variation_values_list)
            unit_name = np.array(unit_name)
            fig, axes = plt.subplots(nrows=1, ncols=self.number_of_varied_variables, sharey=True, figsize=(10, 6))
            for ii, ax in enumerate(axes):
                ax.scatter(float_variation_values_list[ii], ordinate, color='darkblue')
                ax.set(xlabel=(self.varied_variable[ii] + ', ' + unit_name[ii]), yscale=scale)
                ax.minorticks_on()
                ax.grid(which='both')
            axes[0].set(ylabel=value)
        if saveas is not None:
            fig.savefig(saveas+'.jpg', dpi=600, bbox_inches='tight')
            
    def plot_correlation(self, value_x: str, value_y: str, saveas: str = None):
        """Plots value_x against value_y results for all the variations from Optimetrics

        Args:
            value_x (str): the value to plot along x-axis. Copy from EPR_master.print_variations() row titles.
            value_y (str): the value to plot along y-axis. Copy from EPR_master.print_variations() row titles.
            saveas (str, optional): name of the file to save (without extension). If None, file won't be saved.
        """
        try:
            _ = self.variation_values_list
        except AttributeError:
            self.print_variations()
        try:
            abscissa = self.results[value_x]
        except KeyError:
            raise KeyError(f'There is no data field \"{value_x}\". Check if you copied it from EPR_master.print_variations() function correctly.')
        try:
            ordinate = self.results[value_y]
        except KeyError:
            raise KeyError(f'There is no data field \"{value_y}\". Check if you copied it from EPR_master.print_variations() function correctly.')
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.scatter(abscissa, ordinate, color='red')
        ax.set(ylabel=value_y, xlabel=value_x)
        ax.minorticks_on()
        ax.grid(which='both')
        if saveas is not None:
            fig.savefig(saveas+'.jpg', dpi=600, bbox_inches='tight')
            
    def _retrieve_variation(self, var_index: str) -> collections.OrderedDict:
        """Retrieves the analysis of the variation (var_index) from class data. Analyses the variation if it was not done before.

        Args:
            var_index (str): the index of the variation to retrieve.

        Returns:
            collections.OrderedDict: container formed by pyEPR containing Hamiltonian parameters of the model.
        """
        try:
            _int_var_index = int(var_index)
        except ValueError:
            raise ValueError("Variation index should be an integer number. Watch yourself next time")
        else:
            if self.pyEPR_results[_int_var_index] is None:
                pyEPR_result = self.epra.analyze_variation(variation=var_index, cos_trunc=6, fock_trunc=7, modes=self.modes_indices, print_result=False)
                pyEPR_result['f_0'], pyEPR_result['Qs'] = map(self._make_proper_pandas_axis, [pyEPR_result['f_0'], pyEPR_result['Qs']])
                self.pyEPR_results[_int_var_index] = pyEPR_result
                self.losses_discarded[_int_var_index] = check_if_all_same(pyEPR_result['Qs'], float('inf'))
            else:
                pyEPR_result = self.pyEPR_results[_int_var_index]
        return pyEPR_result
    
    def _make_proper_pandas_axis(self, pd_data_series: pd.Series) -> pd.Series:
        """Rename axes such that their names match for pandas Series directly accessed from HFSS (self.modes_indices) and those created by pyEPR ([0, 1, 2...])

        Args:
            pd_data_series (pd.Series): pandas Series to change axis to [0, 1, 2...]

        Returns:
            pd.Series: same pandas Series with renamed axes
        """
        return pd_data_series.set_axis(list(range(self.number_of_modes)))
            
def create_mode_names(sample: np.ndarray, inds: np.ndarray, name: str) -> np.ndarray:
    """Updates modes names: for the indices inds of a sample replaces the value to name_num where num is a number

    Args:
        sample (np.ndarray): default modes names
        inds (np.ndarray): indices of name mode
        name (str): name of the mode

    Returns:
        np.ndarray: updated modes names
    """
    if len(inds) == 0:
        return sample
    upd_sample = sample
    if len(inds) == 1:
        upd_sample[inds[0]] = name
    else:
        for num, idx in enumerate(inds):
            upd_sample[idx] = (" ".join([name, str(num+1)]))
    return upd_sample

def check_if_all_same(list_of_elem: list, to_check = None) -> bool:
    """ Check if all values in list are equal to to_check
    
    Args:
        list_of_elem (list) - list to check for being all to_check.
        to_check (optional) - the value to check. Defaults to None
        
    Returns:
        bool - True if all th elements of the list are None, otherwise False.
    """
    result = True
    if to_check is None:
        for elem in list_of_elem:
            if elem is not None:
                return False
    else:
        for elem in list_of_elem:
            if elem != to_check:
                return False
    return result

def get_modes_list(dict_of_all_modes: dict, modes_to_analyse: set = {'Qubit', 'RR', 'Cavity'}) -> dict:
    """If one intends to analyse only some of the modes solved by HFSS, one can use this function to get ordered dict of the indices of the needed modes

    Args:
        dict_of_all_modes (dict): names and indices of all the modes solved by HFSS
        modes_to_analyse (set, optional): names of the modes one wants to analyse with pyEPR. Defaults to {'Qubit', 'RR', 'Cavity'}.

    Returns:
        dict: dictionary of the format {"Mode": index} with only needed modes, ordered by indices (to provide to pyEPR routine)
    """
    _modes_to_analyse = copy.deepcopy(dict_of_all_modes)
    for _mode_name in dict_of_all_modes.keys():
        if _mode_name not in modes_to_analyse:
            del _modes_to_analyse[_mode_name]
    return _modes_to_analyse