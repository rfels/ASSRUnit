####################################################################
#
#  only for test purposes
#
#
####################################################################

import os

if 'compiling_done' in locals():
    pass
else :
    compiling_done = False

cwd = os.getcwd()

if compiling_done :
    print("Compiling already done!")
    print("working dir: " + cwd)
else :
    # path to data
    data_path= '/assrunit/acnet'
    os.chdir(cwd+data_path)
    print("working dir changed: " + os.getcwd())

    # start neuron netpyne to compile
    print("START Compiling ")
    os.system('nrnivmodl '+ os.getcwd() + '/mod_files')
    compiling_done = True
    print("FINISHED Compiling")


import sciunit

from assrunit.capabilities import ProduceXY

from assrunit.models import ACNetMinimalModel

from assrunit.tests.test_and_prediction_tests import Test4040,Test3030,Test2020,Test2040,Test4020

# with adapted path in directory
#controlparams = {'tau1_gaba_IE': 0.5,'tau2_gaba_IE': 8.0,'tau1_gaba_II': 0.5,'tau2_gaba_II': 8.0,'nmda_alpha': 10.0,'nmda_beta': 0.015,'nmda_e': 45.0,'nmda_g': 1.0,'nmda_gmax': 1.0,'stim_bkg_pyr_weight': 0.0325, 'stim_bkg_bask_weight': 0.002, 'stim_drive_pyr_weight': 0.1, 'stim_drive_bask_weight': 0.1, 'conn_pyr_pyr_weight': [0.0012,0.0006], 'conn_pyr_bask_weight': [0.0012,0.00013], 'conn_bask_pyr_weight': 0.035, 'conn_bask_bask_weight': 0.023, 'duration': 2, 'frequency': 40, 'directory': os.getcwd(), 'conn_seed_file': 'Conn-Seeds.npy', 'noise_seed_file': 'Noise-Seeds.npy'}
controlparams = {'tau1_gaba_IE': 0.5,'tau2_gaba_IE': 8.0,'tau1_gaba_II': 0.5,'tau2_gaba_II': 8.0,'nmda_alpha': 10.0,'nmda_beta': 0.015,'nmda_e': 45.0,'nmda_g': 1.0,'nmda_gmax': 1.0,'stim_bkg_pyr_weight': 0.0325, 'stim_bkg_bask_weight': 0.002, 'stim_drive_pyr_weight': 0.1, 'stim_drive_bask_weight': 0.1, 'conn_pyr_pyr_weight': [0.0012,0.0006], 'conn_pyr_bask_weight': [0.0012,0.00013], 'conn_bask_pyr_weight': 0.035, 'conn_bask_bask_weight': 0.023, 'duration': 2000, 'frequency': 40,'dt': 0.25, 'directory': os.getcwd(), 'seeds': [123,321,111]}
#schizparams = {'tau1_gaba_IE': 0.5,'tau2_gaba_IE': 28.0,'tau1_gaba_II': 0.5,'tau2_gaba_II': 28.0,'nmda_alpha': 10.0,'nmda_beta': 0.015,'nmda_e': 45.0,'nmda_g': 1.0,'nmda_gmax': 1.0,'stim_bkg_pyr_weight': 0.0325, 'stim_bkg_bask_weight': 0.002, 'stim_drive_pyr_weight': 0.1, 'stim_drive_bask_weight': 0.1, 'conn_pyr_pyr_weight': [0.0012,0.0006], 'conn_pyr_bask_weight': [0.0012,0.00013], 'conn_bask_pyr_weight': 0.035, 'conn_bask_bask_weight': 0.023, 'duration': 2, 'frequency': 40, 'directory': os.getcwd(), 'conn_seed_file': 'Conn-Seeds.npy', 'noise_seed_file': 'Noise-Seeds.npy'}
schizparams = {'tau1_gaba_IE': 0.5,'tau2_gaba_IE': 28.0,'tau1_gaba_II': 0.5,'tau2_gaba_II': 28.0,'nmda_alpha': 10.0,'nmda_beta': 0.015,'nmda_e': 45.0,'nmda_g': 1.0,'nmda_gmax': 1.0,'stim_bkg_pyr_weight': 0.0325, 'stim_bkg_bask_weight': 0.002, 'stim_drive_pyr_weight': 0.1, 'stim_drive_bask_weight': 0.1, 'conn_pyr_pyr_weight': [0.0012,0.0006], 'conn_pyr_bask_weight': [0.0012,0.00013], 'conn_bask_pyr_weight': 0.035, 'conn_bask_bask_weight': 0.023, 'duration': 2000, 'frequency': 40, 'dt': 0.25, 'directory': os.getcwd(), 'seeds': [123,321,111]}

test_model = ACNetMinimalModel(controlparams,schizparams)

test_4040 = Test4040(observation={'ratio':0.5})
score_4040 = test_4040.judge(test_model)
print(score_4040)

test_3030 = Test3030(observation={'ratio':1.0})
score_3030 = test_3030.judge(test_model)
print(score_3030)

test_2020 = Test2020(observation={'ratio':1.0})
score_2020 = test_2020.judge(test_model)
print(score_2020)

test_2040 = Test2040(observation={'ratio':1.0})
score_2040 = test_2040.judge(test_model)
print(score_2040)

test_4020 = Test4020(observation={'ratio':1.0})
score_4020 = test_4020.judge(test_model)
print(score_4020)


ACNet_main_suite = sciunit.TestSuite([test_4040,test_3030,test_2020,test_4020,test_2040])
score_matrix = ACNet_main_suite.judge(test_model)
score_matrix