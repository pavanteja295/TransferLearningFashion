'''
 this code creates plots for Top-1 and Top-5 accuracy
 Json files needs to be generated from main.py --fine_test

'''
import json
import numpy as np
from collections import OrderedDict 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


json_files = {}

# check jsons folder to load the files
json_files['finetune'] = 'jsons/fashion_finetune_lr_0.1_resize_224_yes_pretrained_in_resnet18_original_freeze_b4_layer4_random_resize_crop_weight_decay_5e-4train_40_train_60_finetune_switch_all_40_result.json'
json_files['pretrain'] = 'jsons/fashion_full_pretrain_lr_0.1_resize_224_yes_pretrained_in_resnet18__result.json'

for key_, json_file in json_files.items():
	with open(json_file, 'r') as fp:
		json_ = json.load(fp)


	top_1_cls = json_['acc_cl_1']
	x_ticks = np.array(range(0, len(top_1_cls), 1))
	#import pdb; pdb.set_trace()

	dict_sort_1 = {k: v for k, v in sorted(json_['acc_cl_1'].items(), key=lambda item: item[1][2])}
	dict_sort_5 = {k: v for k, v in sorted(json_['acc_cl_5'].items(), key=lambda item: item[1][2])}
	acc_keys_ = [key_ for key_, val in dict_sort_1.items()]
	acc_1 = [ round(val_[0] * 100, 2)  for key_, val_ in dict_sort_1.items()]
	cls_num = [f'{key_}_{int(val_[2])}'   for key_, val_ in dict_sort_1.items()]
	plt.bar(x_ticks-0.2, acc_1, width=0.4,  align='center', tick_label = cls_num)
	#if key_== 'pretrain':
	acc_5 = [ round(val_[0]* 100, 2) for key_, val_ in dict_sort_5.items()]
	plt.bar(x_ticks + 0.2, acc_5, width=0.4, color= 'orange',align='center', tick_label = cls_num)

	plt.xlim(0, len(x_ticks))
	plt.xticks(rotation=90)
	plt.subplots_adjust(left=0.05, bottom=0.36, right=1.0, top=0.96)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(18.5, 10.5)
	# plt.show()
	print('Saving plots for ', key_ )
	plt.savefig('plots/' + key_ + '.png', dpi=100)
	plt.close()
