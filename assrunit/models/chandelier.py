####################################################################
# Implements an extension of the simple model from Vierling-Claassen et al.,
# J Neurophysiol, 2008
# 
# This extension has two populations of inhibitory neurons: basket cells
# and chandelier cells
#
# @author: Christoph Metzner, 03/02/2017
####################################################################

import sciunit
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from assrunit.capabilities import ProduceXY


class SimpleModelExtended(object):
    """

    The simple model from Vierling-Claassen et al. (J Neurophysiol, 2008)

	Attributes:
		n_ex		: number of excitatory cells
		n_bask		: number of basket cells
       	n_chand    : number of chandelier cells

		eta		:
		tau_R		: 'synaptic rise time'
		tau_ex		: exc. 'synaptic decay time'
		tau_bask	: inh. 'synaptic decay time' for basket cells
  		tau_chand	: inh. 'synaptic decay time' for chandelier cells

		g_ee		: E-E weight
		g_eb		: E-B weight
  		g_ec		: E-C weight
		g_be		: B-E weight
  		g_ce		: C-E weight
		g_bb		: B-B weight
  		g_cc		: C-C weight
    		g_bc		: B-C weight
		g_de		: Drive-E weight
		g_db		: Drive-B weight
  		g_dc		: Drive-C weight

		dt		: time step

		b_ex		: applied current to excitatory cells
		b_bask		: applied current to basket cells
  		b_chand		: applied current to chandelier cells
		drive_frequency : drive frequency


		background_rate : rate of the background noise spike trains
		A		: scaling factor for the background noise strength

		seed		: seed for the random generator
	"""

    def __init__(self, params):
        self.n_ex = params["n_ex"]
        self.n_bask = params["n_bask"]
        self.n_chand = params["n_chand"]
        self.eta = params["eta"]
        self.tau_R = params["tau_R"]
        self.tau_ex = params["tau_ex"]
        self.tau_bask = params["tau_bask"]
        self.tau_chand = params["tau_chand"]
        self.g_ee = params["g_ee"]
        self.g_eb = params["g_eb"]
        self.g_ec = params["g_ec"]
        self.g_be = params["g_be"]
        self.g_ce = params["g_ce"]
        self.g_bb = params["g_bb"]
        self.g_cc = params["g_cc"]
        self.g_bc = params["g_bc"]
        self.g_de = params["g_de"]
        self.g_db = params["g_db"]
        self.g_dc = params["g_dc"]
        self.dt = params["dt"]
        self.b_ex = params["b_ex"]
        self.b_bask = params["b_bask"]
        self.b_chand = params["b_chand"]
        self.background_rate = params["background_rate"]
        self.A = params["A"]
        self.filename = params["filename"]
        self.directory = params["directory"]

    def run(self, drive_frequency=40.0, seed=12345, time=500, saveMEG=0, saveEX=0, saveBASK=0, saveCHAND=0):
        """
        Runs the model and returns (and stores) the results

        Parameters:
        time : the length of the simulation (in ms)
        saveMEG: flag that signalises whether the MEG signal should be stored
        saveEX: flag that signalises whether the exc. population activity should be stored
        saveBASK: flag that signalises whether the basket cell population activity should be stored
        saveCHAND: flag that signalises whether the chandelier cell population activity should be stored
        """

        drive_frequency = drive_frequency
        seed = seed

        time_points = np.linspace(0, time, int(time / self.dt) + 1)  # number of time steps (in ms)

        # Initialisations
        drive_cell = np.zeros((len(time_points),))  # the pacemaking drive cell

        theta_ex = np.zeros((self.n_ex, len(time_points)))  # exc. neurons
        theta_bask = np.zeros((self.n_bask, len(time_points)))  # basket cells
        theta_chand = np.zeros((self.n_chand, len(time_points)))  # basket cells

        s_ee = np.zeros((self.n_ex, self.n_ex, len(time_points)))  # E-E snyaptic gating variables
        s_eb = np.zeros((self.n_ex, self.n_bask, len(time_points)))  # E-B snyaptic gating variables
        s_ec = np.zeros((self.n_ex, self.n_chand, len(time_points)))  # E-C snyaptic gating variables
        s_be = np.zeros((self.n_bask, self.n_ex, len(time_points)))  # B-E snyaptic gating variables
        s_ce = np.zeros((self.n_chand, self.n_ex, len(time_points)))  # C-E snyaptic gating variables
        s_bb = np.zeros((self.n_bask, self.n_bask, len(time_points)))  # B-B snyaptic gating variables
        s_cc = np.zeros((self.n_chand, self.n_chand, len(time_points)))  # C-C snyaptic gating variables
        s_bc = np.zeros((self.n_bask, self.n_chand, len(time_points)))  # B-C snyaptic gating variables
        s_de = np.zeros((self.n_ex, len(time_points)))  # Drive-E snyaptic gating variables
        s_db = np.zeros((self.n_bask, len(time_points)))  # Drive-B snyaptic gating variables
        s_dc = np.zeros((self.n_chand, len(time_points)))  # Drive-C snyaptic gating variables

        N_ex = np.zeros((self.n_ex, len(time_points)))  # Noise to exc. cells
        N_bask = np.zeros((self.n_bask, len(time_points)))  # Noise to basket cells
        N_chand = np.zeros((self.n_chand, len(time_points)))  # Noise to chandelier cells

        S_ex = np.zeros((self.n_ex, len(time_points)))  # Synaptic inputs for exc. cells
        S_bask = np.zeros((self.n_bask, len(time_points)))  # Synaptic inputs for basket cells
        S_chand = np.zeros((self.n_chand, len(time_points)))  # Synaptic inputs for chandelier cells

        meg = np.zeros((self.n_ex, len(time_points)))  # MEG component for each cell (only E-E EPSCs)

        # applied currents
        B_ex = self.b_ex * np.ones((self.n_ex,))  # applied current for exc. cells
        B_bask = self.b_bask * np.ones((self.n_bask,))  # applied current for basket cells
        B_chand = self.b_chand * np.ones((self.n_chand,))  # applied current for chand cells

        # Frequency = 1000/period(in ms) and b= pi**2 / period**2 (because period = pi* sqrt(1/b); see Boergers and Kopell 2003) 
        period = 1000.0 / drive_frequency
        b_drive = np.pi ** 2 / period ** 2  # applied current for drive cell

        # Seed the random generator
        random.seed(seed)

        # Noise spike trains
        ST_ex = [None] * self.n_ex
        ST_bask = [None] * self.n_bask
        ST_chand = [None] * self.n_chand

        rate_parameter = 1000 * (1.0 / self.background_rate)
        rate_parameter = 1.0 / rate_parameter
        for i in range(self.n_ex):
            template_spike_array = []
            # Produce Poissonian spike train
            total_time = 0.0
            while total_time < time:
                next_time = random.expovariate(rate_parameter)
                total_time = total_time + next_time
                if total_time < time:
                    template_spike_array.append(total_time)

            ST_ex[i] = template_spike_array

        for i in range(self.n_bask):
            template_spike_array = []
            # Produce Poissonian spike train
            total_time = 0.0
            while total_time < time:
                next_time = random.expovariate(rate_parameter)
                total_time = total_time + next_time
                if total_time < time:
                    template_spike_array.append(total_time)

            ST_bask[i] = template_spike_array

        for i in range(self.n_chand):
            template_spike_array = []
            # Produce Poissonian spike train
            total_time = 0.0
            while total_time < time:
                next_time = random.expovariate(rate_parameter)
                total_time = total_time + next_time
                if total_time < time:
                    template_spike_array.append(total_time)

            ST_chand[i] = template_spike_array

        a = np.zeros((self.n_ex, 1))
        b = np.zeros((self.n_bask, 1))
        c = np.zeros((self.n_chand, 1))
        # Simulation
        for t in range(1, len(time_points)):
            # calculate noise (not done in a very efficient way!)
            for i in range(self.n_ex):
                for tn in ST_ex[i]:
                    N_ex[i, t] = N_ex[i, t] + self._noise(t, tn)

                    # calculate noise (not done in a very efficient way!)
            for i in range(self.n_bask):
                for tn in ST_bask[i]:
                    N_bask[i, t] = N_bask[i, t] + self._noise(t, tn)

                    # calculate noise (not done in a very efficient way!)
            for i in range(self.n_chand):
                for tn in ST_chand[i]:
                    N_chand[i, t] = N_chand[i, t] + self._noise(t, tn)

                    # evolve gating variables
            s_ee[:, :, t] = s_ee[:, :, t - 1] + self.dt * (-1.0 * (s_ee[:, :, t - 1] / self.tau_ex) + np.exp(
                -1.0 * self.eta * (1 + np.cos(theta_ex[:, t - 1]))) * ((1.0 - s_ee[:, :, t - 1]) / self.tau_R))
            # this seems awfully complicated            
            for k in range(self.n_ex):
                a[k, 0] = theta_ex[k, t - 1]
            s_eb[:, :, t] = s_eb[:, :, t - 1] + self.dt * (
                    -1.0 * (s_eb[:, :, t - 1] / self.tau_ex) + np.exp(-1.0 * self.eta * (1 + np.cos(a))) * (
                    (1.0 - s_eb[:, :, t - 1]) / self.tau_R))
            s_ec[:, :, t] = s_ec[:, :, t - 1] + self.dt * (
                    -1.0 * (s_ec[:, :, t - 1] / self.tau_ex) + np.exp(-1.0 * self.eta * (1 + np.cos(a))) * (
                    (1.0 - s_ec[:, :, t - 1]) / self.tau_R))
            # this seems awfully complicated            
            for l in range(self.n_bask):
                b[l, 0] = theta_bask[l, t - 1]
            s_be[:, :, t] = s_be[:, :, t - 1] + self.dt * (
                    -1.0 * (s_be[:, :, t - 1] / self.tau_bask) + np.exp(-1.0 * self.eta * (1 + np.cos(b))) * (
                    (1.0 - s_be[:, :, t - 1]) / self.tau_R))
            s_bb[:, :, t] = s_bb[:, :, t - 1] + self.dt * (-1.0 * (s_bb[:, :, t - 1] / self.tau_bask) + np.exp(
                -1.0 * self.eta * (1 + np.cos(theta_bask[:, t - 1]))) * ((1.0 - s_bb[:, :, t - 1]) / self.tau_R))
            s_bc[:, :, t] = s_bc[:, :, t - 1] + self.dt * (
                    -1.0 * (s_bc[:, :, t - 1] / self.tau_bask) + np.exp(-1.0 * self.eta * (1 + np.cos(b))) * (
                    (1.0 - s_bc[:, :, t - 1]) / self.tau_R))
            for m in range(self.n_chand):
                c[m, 0] = theta_chand[m, t - 1]
            s_ce[:, :, t] = s_ce[:, :, t - 1] + self.dt * (
                    -1.0 * (s_ce[:, :, t - 1] / self.tau_chand) + np.exp(-1.0 * self.eta * (1 + np.cos(c))) * (
                    (1.0 - s_ce[:, :, t - 1]) / self.tau_R))
            s_cc[:, :, t] = s_cc[:, :, t - 1] + self.dt * (-1.0 * (s_cc[:, :, t - 1] / self.tau_chand) + np.exp(
                -1.0 * self.eta * (1 + np.cos(theta_chand[:, t - 1]))) * ((1.0 - s_cc[:, :, t - 1]) / self.tau_R))

            s_de[:, t] = s_de[:, t - 1] + self.dt * (-1.0 * (s_de[:, t - 1] / self.tau_ex) + np.exp(
                -1.0 * self.eta * (1 + np.cos(drive_cell[t - 1]))) * ((1.0 - s_de[:, t - 1]) / self.tau_R))
            s_db[:, t] = s_db[:, t - 1] + self.dt * (-1.0 * (s_db[:, t - 1] / self.tau_ex) + np.exp(
                -1.0 * self.eta * (1 + np.cos(drive_cell[t - 1]))) * ((1.0 - s_db[:, t - 1]) / self.tau_R))
            s_dc[:, t] = s_dc[:, t - 1] + self.dt * (-1.0 * (s_dc[:, t - 1] / self.tau_ex) + np.exp(
                -1.0 * self.eta * (1 + np.cos(drive_cell[t - 1]))) * ((1.0 - s_dc[:, t - 1]) / self.tau_R))

            # calculate total synaptic input
            S_ex[:, t] = self.g_ee * np.sum(s_ee[:, :, t - 1], axis=0) - self.g_be * np.sum(s_be[:, :, t - 1],
                                                                                            axis=0) - self.g_ce * np.sum(
                s_ce[:, :, t - 1], axis=0) + self.g_de * s_de[:, t - 1]
            S_bask[:, t] = self.g_eb * np.sum(s_eb[:, :, t - 1], axis=0) - self.g_bb * np.sum(s_bb[:, :, t - 1],
                                                                                              axis=0) + self.g_db * s_db[
                                                                                                                    :,
                                                                                                                    t - 1]
            S_chand[:, t] = self.g_ec * np.sum(s_ec[:, :, t - 1], axis=0) - self.g_cc * np.sum(s_cc[:, :, t - 1],
                                                                                               axis=0) - self.g_bc * np.sum(
                s_bc[:, :, t - 1], axis=0) + self.g_dc * s_dc[:, t - 1]

            meg[:, t] = self.g_ee * np.sum(s_ee[:, :, t - 1], axis=0)  # + self.g_de*s_de[:,t-1]

            # evolve drive cell
            drive_cell[t] = drive_cell[t - 1] + self.dt * (
                    (1 - np.cos(drive_cell[t - 1])) + b_drive * (1 + np.cos(drive_cell[t - 1])))

            # evolve theta
            theta_ex[:, t] = theta_ex[:, t - 1] + self.dt * (
                    (1 - np.cos(theta_ex[:, t - 1])) + (B_ex + S_ex[:, t] + N_ex[:, t]) * (
                    1 + np.cos(theta_ex[:, t - 1])))
            theta_bask[:, t] = theta_bask[:, t - 1] + self.dt * (
                    (1 - np.cos(theta_bask[:, t - 1])) + (B_bask + S_bask[:, t] + N_bask[:, t]) * (
                    1 + np.cos(theta_bask[:, t - 1])))
            theta_chand[:, t] = theta_chand[:, t - 1] + self.dt * (
                    (1 - np.cos(theta_chand[:, t - 1])) + (B_chand + S_chand[:, t] + N_chand[:, t]) * (
                    1 + np.cos(theta_chand[:, t - 1])))

        # Sum EPSCs of excitatory cells
        MEG = np.sum(meg, axis=0)

        if saveMEG:
            filenameMEG = self.directory + self.filename + '-MEG.npy'
            np.save(filenameMEG, MEG)

        if saveEX:
            filenameEX = self.directory + self.filename + '-Ex.npy'
            np.save(filenameEX, theta_ex)

        if saveBASK:
            filenameBASK = self.directory + self.filename + '-Bask.npy'
            np.save(filenameBASK, theta_bask)

        if saveCHAND:
            filenameCHAND = self.directory + self.filename + '-Chand.npy'
            np.save(filenameCHAND, theta_chand)

        return MEG, theta_ex, theta_bask, theta_chand

    def plotTrace(self, trace, sim_time, save):
        """
           Plots a trace signal versus time
           Parameters:
           trace: the trace signal to plot
           sim_time: the duration of the simulation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        time = np.linspace(0, sim_time, int(sim_time / self.dt) + 1)
        ax.plot(time, trace, 'k')

        # plt.show()

    def plotMEG(self, MEG, sim_time, save):
        """
           Plots a simulated MEG signal versus time
           Parameters:
           MEG: the simulated MEG signal to plot
           sim_time: the duration of the simulation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        time = np.linspace(0, sim_time, int(sim_time / self.dt) + 1)
        ax.plot(time, MEG, 'k')

        if save:
            filenamepng = self.directory + self.filename + '-MEG.png'
            # print filenamepng
            plt.savefig(filenamepng, dpi=600)

        # plt.show()

    def rasterPlot(self, data, sim_time, save, name):
        """
           Plots a raster plot for an array of spike trains
           Parameters:
           data: array of spike trains
           sim_time: duration of the simulation
        """
        spiketrains = self._getSpikeTimes(data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, times in enumerate(spiketrains):
            y = [i] * len(times)
            ax.plot(times, y, linestyle='None', color='k', marker='|', markersize=10)
            ax.axis([0, sim_time, -0.5, len(spiketrains)])

        if save:
            filenamepng = self.directory + self.filename + '-' + name + '-raster.png'
            # print filenamepng
            plt.savefig(filenamepng, dpi=600)
        # plt.show()

    def calculatePSD(self, meg, sim_time):
        """
           Calculates the power spectral density of a simulated MEG signal
           Parameters:
           meg: the simulated MEG signal
           sim_time: the duration of the simulation
        """
        # fourier sample rate
        fs = 1. / self.dt

        tn = np.linspace(0, sim_time, int(sim_time / self.dt) + 1)

        npts = len(meg)
        startpt = int(0.2 * fs)

        if (npts - startpt) % 2 != 0:
            startpt = startpt + 1

        meg = meg[startpt:]
        tn = tn[startpt:]
        nfft = len(tn)

        pxx, freqs = mlab.psd(meg, NFFT=nfft, Fs=fs, noverlap=0, window=mlab.window_none)
        pxx[0] = 0.0

        return pxx, freqs

    def plotPSD(self, freqs, psd, fmax, save):
        """
            Plots the power spectral density of a simulated MEG signal
            Parameters:
            freqs: frequency vector
            psd: power spectral density vector
            fmax: maximum frequency to display
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(freqs, psd)
        ax.axis(xmin=0, xmax=fmax)

        if save:
            filenamepng = self.directory + self.filename + '-PSD.png'
            # print filenamepng
            plt.savefig(filenamepng, dpi=600)

        return ax

    def _getSingleSpikeTimes(self, neuron):
        """
           Calculates the spike times from the trace of a single theta neuron
           Parameters:
           neuron: the single neuron trace
        """
        spike_times = []
        old = 0.0
        for i, n in enumerate(neuron):

            # if theta passes (2l-1)*pi, l integer, with dtheta/dt>0 then the neuron spikes (see Boergers and Kopell, 2003)
            if (n % (2 * np.pi)) > np.pi and (old % (2 * np.pi)) < np.pi:
                spike_time = i * self.dt
                spike_times.append(spike_time)
            old = n

        return spike_times

    def _getSpikeTimes(self, data):
        '''
           Calculates the spike times from an array of theta neuron traces
           Parameters:
           data: the traces array
        '''
        nx, ny = data.shape
        spike_times_array = [None] * nx
        for i in range(nx):
            spike_times_array[i] = self._getSingleSpikeTimes(data[i, :])

        return spike_times_array

    def _noise(self, t, tn):
        t = t * self.dt
        if t - tn > 0:
            value = (self.A * (np.exp(-(t - tn) / self.tau_ex) - np.exp(-(t - tn) / self.tau_R))) / (
                    self.tau_ex - self.tau_R)
        else:
            value = 0

        return value


class ChandelierSimpleModel(sciunit.Model, ProduceXY):
    """The extended simple chandelier model from Vierling-Claassen et al. (2008) """

    def __init__(
            self,
            controlparams,
            schizparams,
            seed=12345,
            time=500,
            name="ChandelierSimpleModel",
    ):
        self.controlparams = controlparams
        self.schizparams = schizparams
        self.time = time
        self.name = name
        self.seed = seed
        super(ChandelierSimpleModel, self).__init__(
            name=name,
            controlparams=controlparams,
            schizparams=schizparams,
            seed=seed,
            time=time,
        )

    def produce_XY(self, stimfrequency=40.0, powerfrequency=40.0):
        lbound = int((powerfrequency / 2) - 1)
        ubound = int((powerfrequency / 2) + 2)

        # generate the control network and run simulation
        control_model = SimpleModelExtended(self.controlparams)
        print("Control model created")
        control_meg, _, _, _ = control_model.run(
            stimfrequency, self.seed, self.time, 0, 0, 0, 0
        )
        print("Control model simulated")
        control_pxx, freqs = control_model.calculatePSD(control_meg, self.time)
        print("Control PSD calculated")

        # Frequency range from 38-42Hz
        controlXY = np.sum(control_pxx[lbound:ubound])

        # generate the schizophrenia-like network and run simulation
        schiz_model = SimpleModelExtended(self.schizparams)
        print("Schiz model created")
        schiz_meg, _, _, _ = schiz_model.run(stimfrequency, self.seed, self.time, 0, 0, 0, 0)
        print("Schiz model simulated")
        schiz_pxx, freqs = schiz_model.calculatePSD(schiz_meg, self.time)
        print("Schiz PSD calculated")
        # Frequency range from 38-42Hz
        schizXY = np.sum(schiz_pxx[lbound:ubound])

        return [controlXY, schizXY]


class ChandelierSimpleModelRobust(sciunit.Model, ProduceXY):
    """The extended simple chandelier model from Vierling-Claassen et al. (2008) """

    def __init__(self, controlparams, schizparams, seeds, time=500, name=None):
        self.controlparams = controlparams
        self.schizparams = schizparams
        self.time = time
        self.name = name
        self.seeds = seeds
        super(ChandelierSimpleModelRobust, self).__init__(
            name=name,
            controlparams=controlparams,
            schizparams=schizparams,
            time=time,
            seeds=seeds,
        )

    def produce_XY(self, stimfrequency=40.0, powerfrequency=40.0):
        """
        Simulates Y Hz drive to the control and the schizophrenia-like network for all
        random seeds, calculates a Fourier transform of the simulated MEG
        and extracts the power in the X Hz frequency band for each simulation.
        Returns the mean power for the control and the schizophrenia-like network, respectively.
        """

        lbound = (powerfrequency / 2) - 1
        ubound = (powerfrequency / 2) + 2

        controlXY = np.zeros((len(self.seeds),))
        schizXY = np.zeros((len(self.seeds),))

        for i, s in enumerate(self.seeds):
            print("Seed number:", i)
            # generate the control network and run simulation
            control_model = SimpleModelExtended(self.controlparams)
            print("Control model created")
            control_meg, _, _, _ = control_model.run(stimfrequency, s, self.time, 0, 0, 0, 0)
            print("Control model simulated")
            control_pxx, freqs = control_model.calculatePSD(control_meg, self.time)
            print("Control PSD calculated")
            controlXY[i] = np.sum(control_pxx[int(lbound):int(ubound)])

            # generate the schizophrenia-like network and run simulation
            schiz_model = SimpleModelExtended(self.schizparams)
            print("Schiz model created")
            schiz_meg, _, _, _= schiz_model.run(stimfrequency, s, self.time, 0, 0, 0, 0)
            print("Schiz model simulated")
            schiz_pxx, freqs = schiz_model.calculatePSD(schiz_meg, self.time)
            print("Schiz PSD calculated")
            schizXY[i] = np.sum(schiz_pxx[int(lbound):int(ubound)])

        mcontrolXY = np.mean(controlXY)
        mschizXY = np.mean(schizXY)

        return [mcontrolXY, mschizXY]

    def produce_XY_plus(self, stimfrequency=40.0, powerfrequency=40.0):
        """
        Simulates Y Hz drive to the control and the schizophrenia-like network for
        all random seeds, calculates a Fourier transform of the simulated MEG
        and extracts the power in the X Hz frequency band for each simulation.
        Returns the mean power and the power for all individual simulations for the
        control and the schizophrenia-like network, respectively.
        """

        lbound = (powerfrequency / 2) - 1
        ubound = (powerfrequency / 2) + 2

        controlXY = np.zeros((len(self.seeds),))
        schizXY = np.zeros((len(self.seeds),))

        for i, s in enumerate(self.seeds):
            print("Seed number:", i)
            # generate the control network and run simulation
            control_model = SimpleModelExtended(self.controlparams)
            print("Control model created")
            control_meg, _, _, _ = control_model.run(stimfrequency, s, self.time, 0, 0, 0,0)
            print("Control model simulated")
            control_pxx, freqs = control_model.calculatePSD(control_meg, self.time)
            print("Control PSD calculated")
            controlXY[i] = np.sum(control_pxx[int(lbound):int(ubound)])

            # generate the schizophrenia-like network and run simulation
            schiz_model = SimpleModelExtended(self.schizparams)
            print("Schiz model created")
            schiz_meg, _, _, _ = schiz_model.run(stimfrequency, s, self.time, 0, 0, 0, 0)
            print("Schiz model simulated")
            schiz_pxx, freqs = schiz_model.calculatePSD(schiz_meg, self.time)
            print("Schiz PSD calculated")
            schizXY[i] = np.sum(schiz_pxx[int(lbound):int(ubound)])

        mcontrolXY = np.mean(controlXY)
        mschizXY = np.mean(schizXY)

        return [mcontrolXY, mschizXY, controlXY, schizXY]
