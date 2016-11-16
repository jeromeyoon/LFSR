import numpy as np	
import scipy.io as sio

def load_traindata():
	print('Load training data \n')
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_vertical_input.mat')
	train_input_vertical = traindata['LF_input']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_vertical_gt.mat')
	train_gt_vertical = traindata['LF_label']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_horizontal_input.mat')
	train_input_horizontal = traindata['LF_input']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_horizontal_gt.mat')
	train_gt_horizontal = traindata['LF_label']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_4views_input_1.mat')
	train_input_4views_1 = traindata['LF_input']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_4views_gt_1.mat')
	train_gt_4views_1 = traindata['LF_label']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_4views_input_2.mat')
	train_input_4views_2 = traindata['LF_input']
	traindata = sio.loadmat('/research2/iccv2015/HCI_train_4views_gt_2.mat')
	train_gt_4views_2 = traindata['LF_label']
	train_input_4views= np.concatenate([train_input_4views_1,train_input_4views_2],axis=-1)
	train_gt_4views= np.concatenate([train_gt_4views_1,train_gt_4views_2],axis=-1)
	return train_input_vertical,train_input_horizontal,train_input_4views,train_gt_vertical,train_gt_horizontal,train_gt_4views 
	#batch_idxs = (train_input_vertical.shape[-1]+train_input_horizontal.shape[-1],+train_input_4views.shape[-1])/config.batch_size

def load_valdata():
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_vertical_input.mat')
	val_input_vertical = valdata['LF_input']
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_vertical_gt.mat')
	val_gt_vertical = valdata['LF_label']
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_horizontal_input.mat')
	val_input_horizontal = valdata['LF_input']
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_horizontal_gt.mat')
	val_gt_horizontal = valdata['LF_label']
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_4views_input.mat')
	val_input_4views = valdata['LF_input']
	valdata = sio.loadmat('/research2/iccv2015/HCI_val_4views_gt.mat')
	val_gt_4views = valdata['LF_label']
	return val_input_vertial,val_input_horizontal,val_input_4views,val_gt_vertical,val_gt_horizontal,val_gt_4views
