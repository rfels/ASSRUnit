import sciunit

from capabilities2 import ProduceXY


import math,sys,os
import random
import numpy as np

import matplotlib.pylab as plt



from Modules.Vierling import simpleModel

### The model classes ###


class VierlingSimpleModel(sciunit.Model, 
                 ProduceXY):
    """The simple model from Vierling-Claassen et al. (2008) """
    
    def __init__(self, controlparams, schizparams,seed=12345, time=500, name=None): 
        self.controlparams = controlparams
        self.schizparams = schizparams 
	self.time = time
	self.name = name
	self.seed = seed
        super(VierlingSimpleModel, self).__init__(name=name, controlparams = controlparams, schizparams = schizparams,seed=seed,time=time)

    def produce_XY(self,stimfrequency=40.0,powerfrequency=40.0):

	lbound = (powerfrequency/2)-1
	ubound = (powerfrequency/2)+2

	# generate the control network and run simulation
	control_model = simpleModel(self.controlparams)
	print 'Control model created'
	control_meg,_,_ = control_model.run(stimfrequency,self.seed,self.time,0,0,0)
	print 'Control model simulated' 
	control_pxx,freqs = control_model.calculatePSD(control_meg,self.time)
	print 'Control PSD calculated'
	controlXY = np.sum(control_pxx[lbound:ubound]) # Frequency range from 38-42Hz 

	# generate the schizophrenia-like network and run simulation
	schiz_model = simpleModel(self.schizparams)
	print 'Schiz model created'
	schiz_meg,_,_ = schiz_model.run(stimfrequency,self.seed,self.time,0,0,0)
	print 'Schiz model simulated' 
	schiz_pxx,freqs = schiz_model.calculatePSD(schiz_meg,self.time)
	print 'Schiz PSD calculated'
	schizXY = np.sum(schiz_pxx[lbound:ubound]) # Frequency range from 38-42Hz


        return [controlXY,schizXY]



class VierlingSimpleModelRobust(sciunit.Model, ProduceXY):
    """The simple model from Vierling-Claassen et al. (2008) """
    
    def __init__(self, controlparams, schizparams,seeds, time=500, name=None): 
        self.controlparams = controlparams
        self.schizparams = schizparams 
	self.time = time
	self.name = name
	self.seeds = seeds
        super(VierlingSimpleModelRobust, self).__init__(name=name, controlparams = controlparams, schizparams = schizparams,time=time,seeds=seeds)

    def produce_XY(self,stimfrequency=40.0,powerfrequency=40.0):
	'''
	 Simulates Y Hz drive to the control and the schizophrenia-like network for all random seeds, calculates a Fourier transform of the simulated MEG 
	 and extracts the power in the X Hz frequency band for each simulation. Returns the mean power for the control and the schizophrenia-like network, respectively.
	'''

	lbound = (powerfrequency/2)-1
	ubound = (powerfrequency/2)+2


	controlXY = np.zeros((len(self.seeds),))
	schizXY = np.zeros((len(self.seeds),))

	for i,s in enumerate(self.seeds):
		print 'Seed number:',i
		# generate the control network and run simulation
		control_model = simpleModel(self.controlparams)
		print 'Control model created'
		control_meg,_,_ = control_model.run(stimfrequency,s,self.time,0,0,0)
		print 'Control model simulated' 
		control_pxx,freqs = control_model.calculatePSD(control_meg,self.time)
		print 'Control PSD calculated'
		controlXY[i] = np.sum(control_pxx[lbound:ubound]) # Frequency range from 38-42Hz 

		# generate the schizophrenia-like network and run simulation
		schiz_model = simpleModel(self.schizparams)
		print 'Schiz model created'
		schiz_meg,_,_ = schiz_model.run(stimfrequency,s,self.time,0,0,0)
		print 'Schiz model simulated' 
		schiz_pxx,freqs = schiz_model.calculatePSD(schiz_meg,self.time)
		print 'Schiz PSD calculated'
		schizXY[i] = np.sum(schiz_pxx[lbound:ubound]) # Frequency range from 38-42Hz

	mcontrolXY = np.mean(controlXY)
	mschizXY = np.mean(schizXY)

        return [mcontrolXY,mschizXY]

    def produce_XY_plus(self,stimfrequency=40.0,powerfrequency=40.0):
	'''
	 Simulates Y Hz drive to the control and the schizophrenia-like network for all random seeds, calculates a Fourier transform of the simulated MEG 
	 and extracts the power in the X Hz frequency band for each simulation. Returns the mean power and the power for all individual simulations for the 
	 control and the schizophrenia-like network, respectively.
	'''

	lbound = (powerfrequency/2)-1
	ubound = (powerfrequency/2)+2

	controlXY = np.zeros((len(self.seeds),))
	schizXY = np.zeros((len(self.seeds),))

	for i,s in enumerate(self.seeds):
		print 'Seed number:',i
		# generate the control network and run simulation
		control_model = simpleModel(self.controlparams)
		print 'Control model created'
		control_meg,_,_ = control_model.run(stimfrequency,s,self.time,0,0,0)
		print 'Control model simulated' 
		control_pxx,freqs = control_model.calculatePSD(control_meg,self.time)
		print 'Control PSD calculated'
		controlXY[i] = np.sum(control_pxx[lbound:ubound]) # Frequency range from 38-42Hz 

		# generate the schizophrenia-like network and run simulation
		schiz_model = simpleModel(self.schizparams)
		print 'Schiz model created'
		schiz_meg,_,_ = schiz_model.run(stimfrequency,s,self.time,0,0,0)
		print 'Schiz model simulated' 
		schiz_pxx,freqs = schiz_model.calculatePSD(schiz_meg,self.time)
		print 'Schiz PSD calculated'
		schizXY[i] = np.sum(schiz_pxx[lbound:ubound]) # Frequency range from 38-42Hz

	mcontrolXY = np.mean(controlXY)
	mschizXY = np.mean(schizXY)

        return [mcontrolXY,mschizXY,controlXY,schizXY]
