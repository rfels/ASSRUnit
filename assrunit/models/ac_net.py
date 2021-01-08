####################################################################
# Implements the simple model from Vierling-Claassen et al.,
# J Neurophysiol, 2008
#
# @author: Christoph Metzner, 03/02/2017
####################################################################

import sciunit

import numpy as np
from netpyne import sim

import pickle

import os
import assrunit.acnet.ACnet_NMDAparams as ACnet_NMDAparams

from assrunit.capabilities import ProduceXY


class ACNetModel(object):
    """
        Minimal ACNet model

        Attributes:
                tau1_gaba_IE :
                tau2_gaba_IE :
                tau1_gaba_II :
                tau2_gaba_II :
                nmda_alpha :
                nmda_beta :
                nmda_e :
                nmda_g :
                nmda_gmax :
                stim_bkg_pyr_weight :
                stim_bkg_bask_weight :
                stim_drive_pyr_weight :
                stim_drive_bask_weight :
                conn_pyr_pyr_weight :
                conn_pyr_bask_weight :
                conn_bask_pyr_weight :
                conn_bask_bask_weight :
                drive_frequency :

        """

    def __init__(self, params):
        # params
        self.tau1_gaba_IE = params["tau1_gaba_IE"]
        self.tau2_gaba_IE = params["tau2_gaba_IE"]
        self.tau1_gaba_II = params["tau1_gaba_II"]
        self.tau2_gaba_II = params["tau2_gaba_II"]
        self.nmda_alpha = params["nmda_alpha"]
        self.nmda_beta = params["nmda_beta"]
        self.nmda_e = params["nmda_e"]
        self.nmda_g = params["nmda_g"]
        self.nmda_gmax = params["nmda_gmax"]
        self.stim_bkg_pyr_weight = params["stim_bkg_pyr_weight"]
        self.stim_bkg_bask_weight = params["stim_bkg_bask_weight"]
        self.stim_drive_pyr_weight = params["stim_drive_pyr_weight"]
        self.stim_drive_bask_weight = params["stim_drive_bask_weight"]
        self.conn_pyr_pyr_weight = params["conn_pyr_pyr_weight"]
        self.conn_pyr_bask_weight = params["conn_pyr_bask_weight"]
        self.conn_bask_pyr_weight = params["conn_bask_pyr_weight"]
        self.conn_bask_bask_weight_bask_weight = params["conn_bask_bask_weight"]
        self.duration = params["duration"]
        self.freq = params["frequency"]

        self.directory = params["directory"]
        self.conn_seeds_file = params["conn_seed_file"]
        self.noise_seeds_file = params["noise_seed_file"]

    def run(self, name, freq=40):
        #conn_seeds = np.load(self.conn_seeds_file)
        #noise_seeds = np.load(self.noise_seeds_file)

        conn_seeds = [123]  # only for test purposes
        noise_seeds = [123]  # only for test purposes
        # auch hier reicht es einen Seed zu simulieren

        frequencies = [float(freq)]
        # run sims for different subjects
        for i, cs in enumerate(conn_seeds):
            # if i>15:
            print(i)
            # different subject
            directory = self.directory + '/Subject_' + str(i)
            print(directory)
            if not os.path.isdir(directory):
                print('Creating directoy:', directory)
                os.mkdir(directory)
            for j, ns in enumerate(noise_seeds):
                # different trials for each subject
                for f in frequencies:
                    ACnet_NMDAparams.simConfig.filename = directory + '/ACnet_NMDA_' + name + '_trial_' + str(
                        j) + '_' + str(
                        int(f)) + 'Hz'  # Set file output name
                    ACnet_NMDAparams.simConfig.seeds = {'conn': cs, 'stim': ns, 'loc': 1}  # Seeds for randomizers
                    # (connectivity , input stimulation and cell locations)
                    ACnet_NMDAparams.netParams.stimSourceParams['drive'] = {'type': 'NetStim', 'rate': f, 'noise': 0.0,
                                                                            'start': 1000.0}
                    # update params
                    self.update_NMDAparams(self.tau1_gaba_IE, self.tau2_gaba_IE, self.tau1_gaba_II, self.tau2_gaba_II,
                                           self.nmda_alpha, self.nmda_beta, self.nmda_e, self.nmda_g, self.nmda_gmax,
                                           self.stim_bkg_pyr_weight, self.stim_bkg_bask_weight,
                                           self.stim_drive_pyr_weight, self.stim_drive_bask_weight,
                                           self.conn_pyr_pyr_weight, self.conn_pyr_bask_weight,
                                           self.conn_bask_pyr_weight, self.conn_bask_bask_weight_bask_weight,
                                           self.duration)

                    sim.createSimulateAnalyze(netParams=ACnet_NMDAparams.netParams,
                                              simConfig=ACnet_NMDAparams.simConfig)

    '''
    update all NMDA parameters 
    '''

    def update_NMDAparams(self, tau1_gaba_IE, tau2_gaba_IE, tau1_gaba_II, tau2_gaba_II, nmda_alpha, nmda_beta, nmda_e,
                          nmda_g, nmda_gmax, stim_bkg_pyr_weight, stim_bkg_bask_weight, stim_drive_pyr_weight,
                          stim_drive_bask_weight, conn_pyr_pyr_weight, conn_pyr_bask_weight, conn_bask_pyr_weight,
                          conn_bask_bask_weight, duration):
        ACnet_NMDAparams.netParams.synMechParams['GABA_IE']['tau1'] = tau1_gaba_IE
        ACnet_NMDAparams.netParams.synMechParams['GABA_IE']['tau2'] = tau2_gaba_IE
        ACnet_NMDAparams.netParams.synMechParams['GABA_II']['tau1'] = tau1_gaba_II
        ACnet_NMDAparams.netParams.synMechParams['GABA_II']['tau2'] = tau2_gaba_II
        ACnet_NMDAparams.netParams.synMechParams['NMDA']['Alpha'] = nmda_alpha
        ACnet_NMDAparams.netParams.synMechParams['NMDA']['Beta'] = nmda_beta
        ACnet_NMDAparams.netParams.synMechParams['NMDA']['e'] = nmda_e
        ACnet_NMDAparams.netParams.synMechParams['NMDA']['g'] = nmda_g
        ACnet_NMDAparams.netParams.synMechParams['NMDA']['gmax'] = nmda_gmax
        ACnet_NMDAparams.netParams.stimTargetParams['bkg->PYR']['weight'] = stim_bkg_pyr_weight
        ACnet_NMDAparams.netParams.stimTargetParams['bkg->BASK']['weight'] = stim_bkg_bask_weight
        ACnet_NMDAparams.netParams.stimTargetParams['drive->PYR']['weight'] = stim_drive_pyr_weight
        ACnet_NMDAparams.netParams.stimTargetParams['drive->BASK']['weight'] = stim_drive_bask_weight
        ACnet_NMDAparams.netParams.connParams['PYR->PYR']['weight'] = conn_pyr_pyr_weight
        ACnet_NMDAparams.netParams.connParams['PYR->BASK']['weight'] = conn_pyr_bask_weight
        ACnet_NMDAparams.netParams.connParams['BASK->PYR']['weight'] = conn_bask_pyr_weight
        ACnet_NMDAparams.netParams.connParams['BASK->BASK']['weight'] = conn_bask_bask_weight
        ACnet_NMDAparams.simConfig.duration = duration

    def analyze(self, name, freq):
        #conn_seeds = np.load(self.conn_seeds_file)
        #noise_seeds = np.load(self.noise_seeds_file)
        

        frequencies = [float(freq)]

        conn_seeds = [123]  # only for test purposes
        noise_seeds = [123]  # only for test purposes
        # Es reicht hier einfach für einen Seed zu simulieren jeweils. Das verkürzt die Laufzeit sehr stark! Man könnte überlegen auch eine 'Robust' 
        # Klasse zu machen, wie bei den anderen Modellen, bei denen dann alle Seeds genommen werden

        dt = 0.1
        duration = 2000
        timepoints = int((duration / dt) / 2)

        n = len(conn_seeds)
        m = len(noise_seeds)
        lfps = np.zeros((n, m, 1, timepoints))

        print(os.getcwd())

        # Load data
        for i, cs in enumerate(conn_seeds):
            print(i)
            # different subject
            directory = self.directory + '/Subject_' + str(i)
            for j, ns in enumerate(noise_seeds):
                print(j)
                for k, fr in enumerate(frequencies):
                    print(fr)
                    # different trials for eachf subject
                    filename = directory + '/ACnet_NMDA_' + name + '_trial_' + str(j) + '_' + str(int(fr)) + 'Hz.pkl'
                    f = open(filename, 'rb')
                    data = pickle.load(f)
                    lfp = data['simData']['LFP'][10000:20000]
                    lfp = [x[0] for x in lfp]
                    lfps[i, j, k, :] = lfp

        #savefile1 = name + '-Data.npy'
        savefile2 = name + '-Data.npy' # Es reicht hier nur das LFP zu speichern
        #np.save(savefile1, lfps)
        np.save(savefile2, lfps)
        
        # Hier muss jetzt noch das Powerspektrum des LFP berechnet werden und die Power bei 40Hz extrahiert werden. Dies ist dann der Rückgabewert 
        # dieser Funktion
        
        # psd,freqs = mlab.psd(lfp,NFFT=int(timepoints),Fs=1./dt,overlap=0,window=mlab.window_none) sollte funktionieren
        # psd[0] = 0.0
        
        # power4040 = np.sum(psd[38:42]) das extrahiert die Power
        #  return power4040

class ACNetMinimalModel(sciunit.Model, ProduceXY):
    """Minimal ACNet model """

    def __init__(self, controlparams, schizparams):
        """
        Constructor method. Both parameter sets, for the control and the schizophrenia-like
        network, have to be a dictionary containing the following parmaeters
        (Filename,Stimulation Frequency,Random Seed,E-E Weight,I-E Weight,E-I Weight,I-I Weight,
        Background Noise Weight,E-Drive Weight,I-Drive
        Weight,Background Noise Frequency)
        Parameters:
        controlparams: Parameters for the control network
        schizparams: Parameters for the schizophrenia-like network
        name: name of the instance
        """
        print("test model initialized")
        self.controlparams = controlparams
        self.schizparams = schizparams
        super(ACNetMinimalModel, self).__init__(
            controlparams=controlparams, schizparams=schizparams
        )

    def produce_XY(self, stimfrequency=40.0, powerfrequency=40.0):
        """
        Simulates Y Hz drive to the control and the schizophrenia-like network for all
        random seeds, calculates a Fourier transform of the simulated MEG
        and extracts the power in the X Hz frequency band for each simulation.
        Returns the mean power for the control and the schizophrenia-like network, respectively.
        """
        '''drive_period = (
            1.0 / stimfrequency
        ) * 1000'''  # calculate drive period (in ms) from stimulation frequency
        # generate the control network and run simulation
        print("Generating control model")
        control_model = ACNetModel(self.controlparams)
        print("Running control model")
        control_model.run("control")
        # print("Analysing control model")
        # controlXY = control_model.analyse("control")

        # generate the schizophrenia-like network and run simulation
        print("Generating schizophrenia model")
        schiz_model = ACNetModel(self.schizparams)
        print("Running control model")
        schiz_model.run("schiz")
        # print("Analysing control model")
        # schizXY = schiz_model.analyse("schiz")

        # return [controlXY, schizXY]
