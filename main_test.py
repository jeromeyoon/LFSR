import os
import time
from siamese import Model
import tensorflow as tf
from utils import pp,get_image_test 
import scipy.io as sio
import scipy.misc
from scipy.ndimage import gaussian_filter
import numpy as np
import math
import time
import pdb
import glob


flags =tf.app.flags
flags.DEFINE_integer("epochs", 2000000,"Epoch to train")
flags.DEFINE_float("learning_rate",0.0001,"learning_rate for training")
flags.DEFINE_integer("image_wid",32,"cropping size")
flags.DEFINE_integer("image_hei",32,"cropping size")
flags.DEFINE_string("dataset","vertical","the name of training name")
flags.DEFINE_string("checkpoint_dir","checkpoint","the name to save the training network")
flags.DEFINE_string("output","output","The directory name to testset output image ")
flags.DEFINE_integer("batch_size",16,"batch_size")
flags.DEFINE_boolean("is_train",False,"True for training,False for testing")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
	date = time.strftime('%d%m')
	pp.pprint(flags.FLAGS.__flags)
	if not os.path.exists(FLAGS.checkpoint_dir):
        	os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.output):
        	os.makedirs(FLAGS.output)
    	if not os.path.exists(os.path.join('./logs',date)):
		os.makedirs(os.path.join('./logs',date))
    	gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
		if FLAGS.is_train:
        		LFSR = Model(sess,date, image_wid=FLAGS.image_wid,image_hei =FLAGS.image_hei ,batch_size=FLAGS.batch_size,\
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir)
			LFSR.train(FLAGS)
		else:
        		LFSR = Model(sess,date, image_wid=FLAGS.image_wid ,image_hei = FLAGS.image_hei ,batch_size=FLAGS.batch_size,\
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir)
			if LFSR.loadnet(FLAGS.checkpoint_dir):
				print('Load pretrained network \n')
			else:
				print('Fail to Load network \n')

			ssim_val = 0.0
			psnr_val = 0.0
			
			#test_batch_idxs = test_input_vertical.shape[-1]/FLAGS.batch_size
			view =2
			count =0
			if view ==0:
				for tt in [0,5,10,15]:
					for ii in range(5):
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(tt+ii))
						sr1 = inputdata['Predict']
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(ii+tt+5))
						sr2= inputdata['Predict'] 
						final_output =np.zeros((sr1.shape[0],sr2.shape[1],3)).astype(np.float32)
						for ch in range(3):
							tmp1 = np.expand_dims(sr1[:,:,ch],axis=-1)
							tmp2 = np.expand_dims(sr2[:,:,ch],axis=-1)
							tmp = np.concatenate([tmp1,tmp2],axis=-1)
							input_ = np.expand_dims(tmp,axis=0)
							output = LFSR.sess.run(LFSR.output_ver,feed_dict={LFSR.train_input_vertical:input_})
							output = np.squeeze(output)
							final_output[:,:,ch] = output
						sio.savemat(os.path.join('buddha','ang_ver_%04d.mat' %count),{'Predict':final_output})
						count +=1
			if view ==1:
				for tt in [0,5,10,15,20]:
					for ii in range(4):
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(tt+ii))
						sr1 = inputdata['Predict']
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(ii+tt+1))
						sr2= inputdata['Predict'] 
						final_output =np.zeros((sr1.shape[0],sr2.shape[1],3)).astype(np.float32)
						for ch in range(3):
							tmp1 = np.expand_dims(sr1[:,:,ch],axis=-1)
							tmp2 = np.expand_dims(sr2[:,:,ch],axis=-1)
							tmp = np.concatenate([tmp1,tmp2],axis=-1)
							input_ = np.expand_dims(tmp,axis=0)
							output = LFSR.sess.run(LFSR.output_hor,feed_dict={LFSR.train_input_horizontal:input_})
							output = np.squeeze(output)
							final_output[:,:,ch] = output
						sio.savemat(os.path.join('buddha','ang_hor_%04d.mat' %count),{'Predict':final_output})
						count +=1
			if view ==2:
				for tt in [0,5,10,15]:
					for ii in range(4):
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(tt+ii))
						sr1 = inputdata['Predict']
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(tt+ii+1))
						sr2 = inputdata['Predict']
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(tt+ii+5))
						sr3 = inputdata['Predict']
						inputdata= sio.loadmat('/research2/SPL/spaSR/buddha/sr_%04d.mat' %(ii+tt+6))
						sr4= inputdata['Predict'] 
						final_output =np.zeros((sr1.shape[0],sr2.shape[1],3)).astype(np.float32)
						for ch in range(3):
							tmp1 = np.expand_dims(sr1[:,:,ch],axis=-1)
							tmp2 = np.expand_dims(sr2[:,:,ch],axis=-1)
							tmp3 = np.expand_dims(sr3[:,:,ch],axis=-1)
							tmp4 = np.expand_dims(sr4[:,:,ch],axis=-1)
							tmp = np.concatenate([tmp1,tmp2,tmp3,tmp4],axis=-1)
							input_ = np.expand_dims(tmp,axis=0)
							output = LFSR.sess.run(LFSR.output_views,feed_dict={LFSR.train_input_4views:input_})
							output = np.squeeze(output)
							final_output[:,:,ch] = output
						sio.savemat(os.path.join('buddha','ang_views_%04d.mat' %count),{'Predict':final_output})
						count +=1

def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def psnr(img1,img2):
	mse = np.mean((img1-img2)**2)
	if mse ==0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__=='__main__':
	tf.app.run()
