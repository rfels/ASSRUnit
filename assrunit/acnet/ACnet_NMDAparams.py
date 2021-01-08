"""
ACnet_NMDAparams.py 

netParams is a dict containing a set of network parameters using a standardized structure

simConfig is a dict containing a set of simulation configurations using a standardized structure

Contributors: Christoph Metzner, christoph.metzner@gmail.com, 12/05/2017
"""

from netpyne import specs
import numpy as np
import os


netParams = specs.NetParams()   # object of class NetParams to store the network parameters
simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

###############################################################################
# NETWORK PARAMETERS
###############################################################################


# Population parameters (so far simple HH cells -> change to cells based on hoc files in '../Pyr-Hay' and '../Inh')
netParams.popParams['PYR'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR', 'numCells': 256,  'color': 'blue'} # add dict with params for this pop  (196=16*16)
netParams.popParams['BASK'] = {'cellModel': 'BASK_Vierling', 'cellType': 'BASK', 'numCells': 64,  'color': 'red'} # add dict with params for this pop  (64=8*8)


# Cell parameters
## PYR cell properties

cellRule = netParams.importCellParams(label='PYR', conds= {'cellType': 'PYR', 'cellModel': 'PYR_Hay'},fileName='Cells/fourcompartment.hoc', cellName='fourcompartment')
cellRule['secs']['soma']['vinit'] = -80.0
cellRule['secs']['dend']['vinit'] = -80.0
cellRule['secs']['apic_0']['vinit'] = -80.0
cellRule['secs']['apic_1']['vinit'] = -80.0

## INH cell properties
cellRule = netParams.importCellParams(label='BASK', conds= {'cellType': 'BASK', 'cellModel': 'BASK_Vierling'},fileName='Cells/FS.hoc', cellName='Layer2_basket')



# Synaptic mechanism parameters
netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 3.0, 'e': 0}
netParams.synMechParams['NMDA'] = {'mod': 'NMDA', 'Alpha': 10.0, 'Beta': 0.015, 'e': 45.0, 'g': 1.0, 'gmax':1.0 }
netParams.synMechParams['GABA_IE'] = {'mod': 'Exp2Syn', 'tau1': 0.5, 'tau2': 8.0, 'e': -75.0}
netParams.synMechParams['GABA_II'] = {'mod': 'Exp2Syn', 'tau1': 0.5, 'tau2': 8.0, 'e': -75.0}


# Stimulation parameters
netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 200, 'noise': 1.0, 'start': 500} # background noise
netParams.stimTargetParams['bkg->PYR'] = {'source': 'bkg', 'conds': {'popLabel': 'PYR'}, 'weight': 0.0325}#
netParams.stimTargetParams['bkg->BASK'] = {'source': 'bkg', 'conds': {'popLabel': 'BASK'}, 'weight': 0.002}#

netParams.stimSourceParams['drive'] = {'type':'NetStim', 'rate': 40.0,'noise': 0.0, 'start': 1000.0}
netParams.stimTargetParams['drive->PYR'] = {'source': 'drive', 'conds':{'popLabel':'PYR'}, 'weight': 0.1}#0.1}
# select 65% of inhibitory cells (see Vierling-Claassen et al., 2008)
#inhCells = range(64)
#nInh = round(0.65*64)
#cellList = np.random.choice(inhCells,size=nInh,replace=False) 
cellList = np.load('Inh_CellList.npy')  
#print cellList
#np.save('Inh_CellList.npy',cellList)
netParams.stimTargetParams['drive->BASK'] = {'source': 'drive', 'conds':{'popLabel':'BASK','cellList': cellList}, 'weight': 0.1}

# Connectivity parameters
netParams.connParams['PYR->PYR'] = {
    'preConds': {'popLabel': 'PYR'}, 'postConds': {'popLabel': 'PYR'},
    'probability': '0.06',
    'weight': [0.0012,0.0006],  # weight for NMDA synapse 50% of AMPA synapse (matches ~45% as in other model (Spencer,2009) and experiments (Myme et al.,2003))
    'threshold': 10,                    				
    'synMech': ['AMPA','NMDA']} 							

netParams.connParams['PYR->BASK'] = {
    'preConds': {'popLabel': 'PYR'}, 'postConds': {'popLabel': 'BASK'},
    'probability': '0.43',
    'weight': [0.0012,0.00013],  # weight for NMDA synapse 10% of AMPA synapse (matches 10% as in other model (Spencer,2009) and experiments (Munoz et al., 1999;Gonzalez-Burgos et al., 2005))	
    'threshold': 10,                    				
    'synMech': ['AMPA','NMDA']} 							# excitatory synapse

netParams.connParams['BASK->PYR'] = {
    'preConds': {'popLabel': 'BASK'}, 'postConds': {'popLabel': 'PYR'},
    'probability': '0.44',
    'weight': 0.035,
    'threshold': 10,                    				
    'synMech': 'GABA_IE'} 							

netParams.connParams['BASK->BASK'] = {
    'preConds': {'popLabel': 'BASK'}, 'postConds': {'popLabel': 'BASK'},
    'probability': '0.51',
    'weight': 0.023,                    				
    'threshold': 10,                    				
    'synMech': 'GABA_II'}



###############################################################################
# SIMULATION PARAMETERS
###############################################################################

# Simulation parameters
simConfig.hParams['celsius'] = 30.0
#simConfig.duration = 2000 # Duration of the simulation, in ms
# shorter duration
simConfig.duration = 2 # Duration of the simulation, in ms
simConfig.dt = 0.025 # Internal integration timestep to use
simConfig.seeds = {'conn': 1, 'stim': 1, 'loc': 1} # Seeds for randomizers (connectivity, input stimulation and cell locations)
simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
simConfig.verbose = False  # show detailed messages 
simConfig.hParams['cai0_ca_ion'] = 0.0001
simConfig.printPopAvgRates = True
#simConfig.verbose = True

# Recording 
simConfig.recordCells = ['all']  # which cells to record from
#simConfig.recordTraces = {'Vsoma':{'sec':'soma','loc':0.5,'var':'v'}}#,'AMPA':{'sec':'dend','loc':0.5,'var':'AMPA','conds':{'cellType':'PYR'}}}
#simConfig.recordStim = True  # record spikes of cell stims
simConfig.recordStep = 0.1 # Step size in ms to save data (eg. V traces, LFP, etc)
simConfig.recordLFP = [[50,0,50]]



# Saving
simConfig.filename = 'Control_Network/Only_E_Drive/ACnet_NMDA_Test_Only_E_Drive'  # Set file output name
simConfig.saveFileStep = 1000 # step size in ms to save data to disk
simConfig.savePickle = True # Whether or not to write spikes etc. to a .pkl file

# Analysis and plotting 
#simConfig.analysis['plotRaster'] = {'saveFig':True}  # Plot raster
#simConfig.analysis['plotTraces'] = {'include': [100,280],'saveFig':True}  # Plot raster
#simConfig.analysis['plot2Dnet'] = {'view':'xz','showConns':False} # Plot 2D net cells and connections


