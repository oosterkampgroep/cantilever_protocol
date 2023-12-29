""" Python translation and refactoring of the Mathematica code used to model the interference experiment
Equations employed below are from "Zenith of the Quantum Doctrine:
 Dissection of the uses and misuses of quantum theory in the quest for macroscopic mechanical quanta"
N.B: Some equations have been rewritten for better precision

ATTENTION: > PI-PHASE SHIFT/ SUPERPOSITION BRANCH MISSING
           > CODE REVIEW MUST TAKE PLACE BEFORE MADE PUBLIC

User instructions: adjust the physical parameters below and run the script

Author: Joao Machado
Date: 07-06-20
"""

import numpy as np
from scipy import integrate
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Physical parameters; choice of global variables was to avoid excessive variables in functions

Omega: 'angular mechanical frequency 2*pi f in Hertz' = 2 * np.pi * 3000
Lambda: 'freq. dif. between the bare and the shifted mechanical frequencies in Hertz' = 2 * np.pi * 0.045
N_thermal: 'phonon thermal bath occupancy' = 0  # you can change it to temperature and compute N_thermal from it instead
Q: 'mechanical quality factor (f/Gamma)' = np.inf  # 1e5
T2: 'dephasing rate in s' = 1e12  # 0.1
beta: 'amplitude of cantilever motion in ZPM units' = 1e5
N_imp: 'initial phonon number uncertainty; assumed = to imprecision noise' = 0  # 760
nc: 'number of cycles between each pulse' = 5
epsilon: 'qubit angular frequency in Hertz' = 0  # 2 * np.pi * 6e9  # I don't recall the value; please check later

# Auxiliary parameters
Delta = np.sqrt(0.5 * (N_imp + 0.5))  # position uncertainty of the initial mechanical Gaussian state in ZPM units
Tau = 2 * np.pi * nc / Omega  # time lapse between cycles; total duration of the experiment = 3Tau
Gamma = Omega / (2 * np.pi * Q)  # mechanical damping rate
Xi = np.sqrt(Gamma ** 2 - 4 * Lambda ** 2 + 8j * Gamma * Lambda * (N_thermal + 0.5))  # aux. param; cf. thesis eq.(3.71)
hM = (Xi - Gamma) / (2 * Gamma * (N_thermal + 0.5) + 1j * Lambda)  # aux. param; cf. thesis eq.(3.71)
hm = (-Xi - Gamma) / (2 * Gamma * (N_thermal + 0.5) + 1j * Lambda)  # aux. param; cf. thesis eq.(3.71)


class GaussianSuperposition:
    # class for Gaussian superpositions:
    # the superposition is written as a sum of Gaussian states of the form exp(g|a|^2+ka+qa*+c),
    # with a = x+ip; for computational economy, only the coefficients [g,k,q,c] for each function are used
    # the structure of the Gaussian superpositions is [[W_uu],[W_dd],[W_ud]], where {W_jj} are arrays
    # with the coefficients for each Gaussian peak and {'u','d'} label the spin-up and spin-down coefficients
    # The choice of an array list is because the number of Gaussian peaks in each W_jj does not need to be the same,
    # and using arrays for each set of peaks of W_jj eases the control on the indices
    # a check for negative probabilities was not carried out because the peaks themselves can be negative,
    # and even complex, as long as the final sum remains positive

    def __init__(self, coef_list):
        # coef_list is a [n x 4, m x 4, l x 4] list where n, m, l are the number of Gaussian peaks for W_uu, W_dd, W_ud;

        if len(coef_list) != 3:
            raise Exception("State does not have the structure [[W_uu],[W_dd],[W_ud]].\n" +
                            "Input state has list length = ", len(coef))

        uu_coefarray = np.asarray(coef_list[0])  # in case of input as lists; 'shape' attribute is necessary
        dd_coefarray = np.asarray(coef_list[1])
        ud_coefarray = np.asarray(coef_list[2])
        coef_list = [uu_coefarray, dd_coefarray, ud_coefarray]
        # cannot do np.asarray(coef_list) because peak number may differ for each entry

        entry_names = ['uu', 'dd', 'ud']
        for index, w_jj in enumerate(coef_list):
            w_shape = w_jj.shape
            if w_shape[1] != 4 or len(w_shape) != 2:  # w_shape must be (n,4)
                raise Exception("Input for entry W_" + entry_names[index] +
                                " does not have the n x 4 required shape." + "Input has shape ", w_shape)

        self.uu = uu_coefarray
        self.dd = dd_coefarray
        self.ud = ud_coefarray

    @staticmethod
    def peak_info(peak):
        # auxiliary function providing information about the Gaussian peak
        g = peak[0]
        k = peak[1]
        q = peak[2]
        c = peak[3]

        # remove the artificial no_peak instances from the list
        if np.isfinite(c):
            peak_magnitude = np.abs(np.exp(c - k * q / g) * np.sqrt(-np.pi / g))
            peak_position = - np.real((k + q) / (2 * g))
            peak_width = np.sqrt(-1 / (2 * np.real(g)))
            peak_phase = np.imag(c - k * q / g - 0.5 * np.log(-g)) % (2 * np.pi)
        else:
            peak_width = np.inf
            peak_position = 'None'
            peak_magnitude = 0.0
            peak_phase = 'Undefined'

        return peak_magnitude, peak_position, peak_width, peak_phase

    def diagnostics(self, mode='informative'):
        # diagnostics function to monitor the quantum state

        if mode == 'plain':
            print('uu coefficients =', self.uu)
            print('dd coefficients =', self.dd)
            print('ud coefficients =', self.ud)

        elif mode == 'informative':

            logging.info('\nCoefficients for the (spin-up,spin-up) component:')
            for peak in self.uu:
                peak_magnitude, peak_position, peak_width, peak_phase = self.peak_info(peak)
                if peak_magnitude > 1e-10:  # ignore the fast decaying peaks
                    logging.info('Peak at x = %3.3f,  with phase = %1.3f, and magnitude = %3.3f', peak_position,
                                 peak_phase, peak_magnitude)

            logging.info('\nCoefficients for the (spin-down,spin-down) component:')
            for peak in self.dd:
                peak_magnitude, peak_position, peak_width, peak_phase = self.peak_info(peak)
                if peak_magnitude > 1e-10:
                    logging.info('Peak at x = %3.3f,  with phase = %1.3f, and magnitude = %3.3f', peak_position,
                                 peak_phase, peak_magnitude)

            logging.info('\nCoefficients for the (spin-up,spin-down) component:')
            for peak in self.ud:
                peak_magnitude, peak_position, peak_width, peak_phase = self.peak_info(peak)
                if peak_magnitude > 1e-10:
                    logging.info('Peak at x = %3.3f,  with phase = %1.3f, and magnitude = %3.3f', peak_position,
                                 peak_phase, peak_magnitude)

            logging.info('\nState normalisation = %1.5f\n', self.state_norm())

    # state_norm was not done above to avoid redundancy whenever they are used after instantiation
    def state_norm(self):
        # Check if the state is normalisable and computes the normalisation of the full Gaussian superposition
        # integral of the Gaussian over entire phase space is -pi*exp(c-kq/g)/g
        # c.f. https://en.wikipedia.org/wiki/Gaussian_function#Integral_of_a_Gaussian_function

        norm = 0
        diag_coef = np.concatenate((self.uu, self.dd))  # merge the diagonal entries
        for k in diag_coef:  # check diagonal entries
            if k[0] >= 0:
                print('Gaussian width <0 (width =', -1 / k[0], ')')
                raise Exception("State is not normalisable")
            else:
                # g<0 (k[0]<0), so -= contributes with a positive amount
                norm -= np.exp(k[3] - k[1] * k[2] / k[0]) / k[0]

        # since coefficients are complex, norm is also complex
        if np.abs(np.imag(norm)) > 1e-12:
            raise Exception('State norm should be real valued. Norm = ', np.pi * norm)

        norm = np.real(norm)  # if imaginary part = 0, then take just the real part
        if norm <= 0:
            raise Exception('State norm should be positive. Norm = ', np.pi * norm)

        return np.pi * norm

    # Functions for the time-evolution; essentially eqs. 3.63 to 3.70 from the thesis
    # Here 'u' and 'd' refer to the 'up' and 'down' spin states
    @staticmethod
    def dif(t, g0):
        # diffusion coefficient for the diagonal elements W_uu and W_dd
        # :param t: time (double)
        # :param g0: inverse of Gaussian width at t=0 (double)
        # Returns: inverse of Gaussian width after time t (double)

        return g0 * (N_thermal + 0.5) * (np.exp(-Gamma * t) - 1) + np.exp(-Gamma * t)

    @staticmethod
    def dif_ud(t, g0):
        # diffusion coefficient for the antidiagonal element W_ud
        # :param t: time (double)
        # :param g0: inverse of Gaussian width at t=0 (double)
        # Returns: inverse of Gaussian width after time t (double)

        return (g0 * (np.exp(-Xi * t) - 1) + hM - hm * np.exp(-Xi * t)) / (hM - hm)

    # auxiliary coefficients for the Wigner functions; cf. eqs.(3.64-3.70) from the thesis
    # the time evolution is implemented via an update of the coefficients of the Wigner function

    def update_coef_uu(self, t, coef):
        # :param t: time (double)
        # :param coef: coefficient array [g,k,q,c] for a given Gaussian peak at t=0 and density matrix up, up component
        # Gaussian peak has form exp(g|a|^2+ka+qa*+c) , with a = x+ip;
        # Returns: updated coefficients with value at time t (array)

        dif_term = self.dif(t, coef[0])
        guu = coef[0] / dif_term
        kuu = (coef[1] * np.exp(-0.5 * Gamma * t + 1j * (Omega + Lambda) * t)) / dif_term
        quu = (coef[2] * np.exp(-0.5 * Gamma * t - 1j * (Omega + Lambda) * t)) / dif_term
        cuu = coef[3] - np.log(dif_term) + coef[2] * coef[1] * (N_thermal + 0.5) * (1 - np.exp(-Gamma * t)) / dif_term

        return np.array([guu, kuu, quu, cuu])

    def update_coef_dd(self, t, coef):
        # :param t: time (double)
        # :param coef: coefficient array [g,k,q,c] for a given Gaussian peak at t=0 and density matrix (down,down)
        # Gaussian peak has form exp(g|a|^2+ka+qa*+c) , with a = x+ip;
        # Returns: updated coefficients with value at time t (array)

        dif_term = self.dif(t, coef[0])
        gdd = coef[0] / dif_term
        kdd = (coef[1] * np.exp(-0.5 * Gamma * t + 1j * (Omega - Lambda) * t)) / dif_term
        qdd = (coef[2] * np.exp(-0.5 * Gamma * t - 1j * (Omega - Lambda) * t)) / dif_term
        cdd = coef[3] - np.log(dif_term) + coef[2] * coef[1] * (N_thermal + 0.5) * (1 - np.exp(-Gamma * t)) / dif_term
        return np.array([gdd, kdd, qdd, cdd])

    def update_coef_ud(self, t, coef):
        # :param t: time (double)
        # :param coef: coefficient array [g,k,q,c] for a given Gaussian peak at t=0 and density matrix (up,down)
        # Gaussian peak has form exp(g|a|^2+ka+qa*+c) , with a = x+ip;
        # Returns: updated coefficients with value at time t (array)

        dif_term = self.dif_ud(t, coef[0])
        gud = (coef[0] * (hM * np.exp(-Xi * t) - hm) + hM * hm * (1 - np.exp(-Xi * t))) / (dif_term * (hM - hm))
        kud = coef[1] * np.exp(-0.5 * Xi * t + 1j * Omega * t) / dif_term
        qud = coef[2] * np.exp(-0.5 * Xi * t - 1j * Omega * t) / dif_term

        simple_dephasing = coef[3] + (1j * (Lambda - epsilon) + (Gamma - Xi) / 2 - 1 / T2) * t
        cud = simple_dephasing - np.log(dif_term) + coef[2] * coef[1] * (1 - np.exp(-Xi * t)) / (dif_term * (hM - hm))

        return np.array([gud, kud, qud, cud])

    def evolve(self, t):
        # evaluates the time-evolution of the superposition state by updating the coefficients
        # :param t: time (double)

        # W_uu update
        for j, branch in enumerate(self.uu):
            self.uu[j] = self.update_coef_uu(t, branch)

        # W_dd update
        for j, branch in enumerate(self.dd):
            self.dd[j] = self.update_coef_dd(t, branch)

        # W_ud update
        for j, branch in enumerate(self.ud):
            self.ud[j] = self.update_coef_ud(t, branch)

    def pi_over2_pulse(self):
        # performs the pi/2 pulse on the system:
        # W_{u,u} -> 1/2 (W_{u,u} + W_{d,d}) +i/2 (W_{u,d} - W_{d,u})
        # W_{d,d} -> 1/2 (W_{u,u} + W_{d,d}) -i/2 (W_{u,d} - W_{d,u})
        # W_{u,d} -> 1/2 (W_{u,d} + W_{d,u}) +i/2 (W_{u,u} - W_{d,d})
        # pre-factors lead to the extra terms of log(2) and i*pi/2 in the coefficients' list
        # N.B: this method enlarges the W_jj's arrays everytime is called
        # due to the mixing of coefficients in a pi/2 pulse
        # As there is no search and merging of peaks at the same point in phase space,
        # the interference is not immediately visible at the array level

        onehalf_factor = -np.log(2)
        pi_over2_phase = 1j * np.pi / 2
        diag_coef = np.concatenate((self.uu, self.dd))

        # updates uu coefficients after pi/2 pulse
        new_uucoef_list = []
        for branch in diag_coef:
            new_branch = deepcopy(branch)  # deepcopy is needed because uu, dd and ud should not change
            # till all coefficients are copied and combined, in all loops
            new_branch[3] += onehalf_factor
            new_uucoef_list.append(new_branch)

        for branch in self.ud:
            ud_branch = deepcopy(branch)
            du_branch = np.conj(ud_branch)
            ud_branch[3] += onehalf_factor + pi_over2_phase
            du_branch[3] += onehalf_factor - pi_over2_phase
            new_uucoef_list.append(ud_branch)
            new_uucoef_list.append(du_branch)

        # updates dd coefficients after pi/2 pulse
        new_ddcoef_list = []
        for branch in diag_coef:
            new_branch = deepcopy(branch)
            new_branch[3] += onehalf_factor
            new_ddcoef_list.append(new_branch)

        for branch in self.ud:
            ud_branch = deepcopy(branch)
            du_branch = np.conj(ud_branch)
            ud_branch[3] += onehalf_factor - pi_over2_phase
            du_branch[3] += onehalf_factor + pi_over2_phase
            new_ddcoef_list.append(ud_branch)
            new_ddcoef_list.append(du_branch)

        # updates ud coefficients after pi/2 pulse
        new_udcoef_list = []
        for branch in self.uu:
            new_branch = deepcopy(branch)
            new_branch[3] += onehalf_factor + pi_over2_phase
            new_udcoef_list.append(new_branch)

        for branch in self.dd:
            new_branch = deepcopy(branch)
            new_branch[3] += onehalf_factor - pi_over2_phase
            new_udcoef_list.append(new_branch)

        for branch in self.ud:
            ud_branch = deepcopy(branch)
            du_branch = np.conj(ud_branch)
            ud_branch[3] += onehalf_factor
            du_branch[3] += onehalf_factor
            new_udcoef_list.append(ud_branch)
            new_udcoef_list.append(du_branch)

        self.uu = np.array(new_uucoef_list)
        self.dd = np.array(new_ddcoef_list)
        self.ud = np.array(new_udcoef_list)

    @staticmethod
    def gaussian_position(x, coef):
        # Auxiliary Gaussian function to evaluate the marginal distribution in X of Gaussian state exp(g|a|^2+ka+qa*+c)
        # with a = x + 1j * p; Integrating in p leads to sqrt(pi/(-g)) * exp(g*(x + (k+q)/(2g))^2 + c - k*q/g)
        # c.f. https://en.wikipedia.org/wiki/Gaussian_function#Integral_of_a_Gaussian_function
        # use of explicit analytical formula was chosen to avoid numerical integration errors associated
        # with the use of limited phasespace intervals and for speed
        # (no need to evaluate the state over the entire phasespace and integrate over the momentum afterwards)
        # :param x: list or array of real space points to be evaluated
        # :param coef: coefficient array for a single peak; coef = [g, k, q, c]
        # Returns: list with the value of the probability of finding the system at every point x for a single Gaussian

        g = coef[0]
        k = coef[1]
        q = coef[2]
        c = coef[3]
        peak_position = -(k + q) / (2 * g)
        peak_magnitude = c - k * q / g  # - 0.5 * np.log(-g) + 0.5 * np.log(np.pi)

        return np.exp(g * (x - peak_position) ** 2 + peak_magnitude) * np.sqrt(-np.pi / g)

    def prob_x(self, plot=False):
        # evaluates the probability of finding the system at each point x
        # :param view_interval: optional subset of points for the position (list)
        # :param plot: optional marker in case we wish to plot the distribution (boolean)
        # Returns: probability distribution evaluated at each point in position_interval (list)

        # find the peak positions, so the probability function is evaluated at the right points
        position_interval = []
        nr_points_side = 60  # number of points on each side of the peak
        for peak in np.concatenate((self.uu, self.dd)):
            peak_magnitude, peak_position, peak_width, peak_phase = self.peak_info(peak)
            if peak_magnitude > 1e-10:
                increment = np.real(peak_width / 12)  # good enough to have a smooth peak
                peak_points = [peak_position + increment * (n - nr_points_side) for n in range(nr_points_side * 2 + 1)]
                position_interval += peak_points

        position_interval.sort()
        prob_dist = np.zeros(len(position_interval))
        imag_res = np.zeros(len(position_interval))  # imaginary residue from evaluating complex functions; if ok == 0

        for index, x in enumerate(position_interval):
            for peak in np.concatenate((self.uu, self.dd)):  # only diagonal entries give probabilities
                prob_dist[index] += self.gaussian_position(x, peak).real
                imag_res[index] += self.gaussian_position(x, peak).imag

            if np.abs(imag_res[index]) > 1e-5:  # 1e-5 is ~ the numerical precision we can achieve with this approach
                raise ValueError('Probability density is complex valued: P(' + str(x) + ') = ' + str(prob_dist[index])
                                 + ' + ' + str(imag_res[index]) + 'j')

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            ax.plot(position_interval, prob_dist, color='green', linewidth=1.2, linestyle='-', label="")
            ax.set_xlabel(r'x (ZPM units)', fontsize=17)
            ax.set_ylabel(r'Probability density', fontsize=17)

            xmin = min(position_interval)
            xmax = max(position_interval)
            ymax = 1.1 * max(prob_dist)
            ax.axis([xmin, xmax, 0, ymax])

            plt.show()

        return prob_dist


# create a logfile to monitor each step of the experiment
logging.basicConfig(filename="InterferenceExperimentLog.log", level=logging.INFO, filemode='w', format='%(message)s')


def log_section(message):
    # Auxiliary function to create visually distinguishable sections in the logfile
    # This is done by wrapping the message around '%'
    # :param message: title of the log section (str)
    # Returns: log section with the message (str)

    message_len = len(message)
    separator = '%' * (message_len + 15) + '\n'
    section = '\n' + separator + '%\n%\t' + message + '\n%\n' + separator

    return section


def interference_experiment():
    # Models the interference experiment we propose and evaluates the final probability of finding the resonator at X

    logging.info("Running the interferometry experiment with the parameters:\n")
    logging.info("Mechanical frequency f = (Omega/2pi) = %1.2e Hz\n", Omega / (2 * np.pi))
    logging.info("Mechanical frequency difference (coupling) = %1.2e Hz\n", Lambda / (2 * np.pi))
    logging.info("Phonon thermal bath occupancy = %3.1f \n", N_thermal)
    logging.info("Mechanical quality factor (f/Gamma) = %1.2e \n", Q)
    logging.info("Dephasing time T_2 = %1.2e s\n", T2)
    logging.info("Amplitude of cantilever motion = %1.2e (ZPM units)\n", beta)
    logging.info("Initial phonon number uncertainty = %3.1f \n", N_imp)
    logging.info("Number of cycles between each pulse = %1.1f \n", nc)
    logging.info("Qubit frequency = %1.2e Hz\n", epsilon / (2 * np.pi))
    logging.info(log_section("Start of the interference experiment.\n"))

    # initial conditions for the cantilever: Gaussian state at x=beta and with uncertainty Delta
    # state has the form exp(g|a|^2+ka+qa*+c), with a = x + i p
    g0 = -1 / (2 * Delta ** 2)
    k0 = -1j * beta / (2 * Delta ** 2)
    q0 = np.conj(k0)
    c0 = -beta ** 2 / (2 * Delta ** 2) - np.log(2 * np.pi * Delta ** 2)

    # quantum states are described by the set of peaks composing them
    initial_gaussian = [g0, k0, q0, c0 + 0j]
    # workaround for the GaussianSuperposition class when entries have no peak; there must be a peak for the pulses
    no_peak = [-1, 0 + 0j, 0 + 0j, np.NINF + 0j]

    # initial state
    q_state = GaussianSuperposition([[initial_gaussian], [no_peak], [no_peak]])
    log_record = 'Initial Gaussian state'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # send a pi/2 pulse to create the initial superposition state
    q_state.pi_over2_pulse()
    log_record = 'Apply the 1st pi/2 pulse to create a superposition state.'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # let it evolve in order for the superposition to be physically separated
    q_state.evolve(Tau)
    log_record = 'Free evolution. After ' + str(nc) + ' cycles, the quantum state becomes:'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # send another pi/2 pulse to divide each superposition branch into 2
    q_state.pi_over2_pulse()
    log_record = 'Apply the 2nd pi/2 pulse to subdivide the superposition state.'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # let it evolve again so 2 of the branches meet at the same point
    # no interference because they are entangled with different spin components
    q_state.evolve(Tau)
    log_record = '2nd free evolution. After another ' + str(nc) + ' cycles, the quantum state becomes:'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # send the final pi/2 pulse to produce the desired interference
    q_state.pi_over2_pulse()
    log_record = 'Apply the 3rd and final pi/2 pulse to be have a which-path experiment.'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    # let it evolve for another time interval in order for the superposition branches to meet and interfere
    q_state.evolve(Tau)
    log_record = 'Final free evolution for ' + str(nc) + ' cycles. The state is now:'
    logging.info(log_section(log_record))
    q_state.diagnostics()

    print('Experiment ended successfully.')
    logging.info('\nEnd of experiment.')

    # plot the expect probability distribution for the mechanical resonator
    q_state.prob_x(plot=True)


if __name__ == '__main__':
    interference_experiment()
