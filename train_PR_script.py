from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config

import numpy as np

thresholds  = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
mu_pos_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
mu_neg_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

thresholds  = [0.5,0.6,0.7,0.8,0.9]
mu_pos_list = [0.6,0.7,0.8,0.9,1.0]
mu_neg_list = [0.0,0.1,0.2,0.3,0.4]

training_data = load_data('../training-data/oos_data/ergos/ergos_train.json')
config_fname = 'sample_configs/tf_test_georg_base.yml'

NT = len(thresholds)
NP = len(mu_pos_list)
NN = len(mu_neg_list)

precisions = np.zeros((NT, NP, NN))
recalls = np.zeros((NT, NP, NN))
F1s = np.zeros((NT, NP, NN))

count = 1
total_num = NT*NP*NN

for intex_t, t in enumerate(thresholds):
	for intex_p, mup in enumerate(mu_pos_list):
		for intex_n, mun in enumerate(mu_neg_list):
			new_config_name = 'sample_configs/param_configs/tf_test_georg_t' + str(t) + '_mp' + str(mup) + '_mn' + str(mun) + '.yml'
			print('working on:')
			print('  threshold = ', str(t))
			print('  mu_pos = ', str(mup))
			print('  mu_neg = ', str(mun))
			print('which is number {} of {} total '.format(count, total_num))
			with open(config_fname) as f:
			    lines = f.readlines()
			    lines = [l for l in lines]
			    lines += ['  out_of_scope_soft_threshold: ' + str(t) + '\n']
			    lines += ['  mu_pos: ' + str(mup) + '\n']
			    lines += ['  mu_neg: -' + str(mun) + '\n']
			    with open(new_config_name, "w") as f1:
			        f1.writelines(lines)
			trainer = Trainer(config.load(new_config_name))
			trainer.train(training_data)
			model_name = 'models/param_models/tf_test_georg_t' + str(t) + '_mp' + str(mup) + '_mn' + str(mun) + '.yml'			
			model_directory = trainer.persist('./projects/default/')




			count += 1