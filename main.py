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
flags.DEFINE_boolean("queue",True,"using multi threads for data loading")
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
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir,queue=FLAGS.queue)
			LFSR.train(FLAGS)
		else:
        		LFSR = Model(sess,date, image_wid=FLAGS.image_wid ,image_hei = FLAGS.image_hei ,batch_size=FLAGS.batch_size,\
	        	dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir,queue=FLAGS.queue)
			if LFSR.loadnet(FLAGS.checkpoint_dir):
				print('Load pretrained network \n')
			else:
				print('Fail to Load network \n')

			testdata = sio.loadmat('/research2/iccv2015/HCI_val_vertical_input.mat')
			test_input_vertical = testdata['LF_input']
			testdata = sio.loadmat('/research2/iccv2015/HCI_val_vertical_gt.mat')
			test_gt_vertical = testdata['LF_label']
			testdata = sio.loadmat('/research2/iccv2015/HCI_val_horizontal_input.mat')
			test_input_horizontal = testdata['LF_input']
			testdata = sio.loadmat('/research2/iccv2015/HCI_val_horizontal_gt.mat')
			test_gt_horizontal = testdata['LF_label']
			testdata = sio.loadmat('/research2/iccv2015/HCI_val_4views_input.mat')
			test_input_4views = testdata['LF_input']
			testdata = sio.loadmat('/research2/iccv2015/HCI_val_4views_gt.mat')
			test_gt_4views = testdata['LF_label']

			ssim_val = 0.0
			psnr_val = 0.0
			
			test_batch_idxs = test_input_vertical.shape[-1]/FLAGS.batch_size
			for ii in range(test_batch_idxs):

				view =0 
				if view ==0:
					batch_files = range(test_input_vertical.shape[-1])[ii*FLAGS.batch_size:(ii+1)*FLAGS.batch_size]	
					batches =[get_image_test(test_input_vertical[0,batch],test_gt_vertical[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					input_ = batches[:,:,:,:2]
					gt = batches[:,:,:,-1]
					start_time = time.time()
					output = LFSR.sess.run(LFSR.output_ver,feed_dict={LFSR.train_input_vertical:input_})
					end_time = time.time()-start_time
					print('Test processing %d/%d time:%f \n' %(ii,test_input_vertical.shape[-1],end_time))
					output = np.squeeze(output)
					gt = np.squeeze(gt)
					ssim_val += ssim_exact(gt.astype(float)/255.0,output)
					psnr_val += psnr(gt.astype(np.uint8),np.uint8(output*255))			
					scipy.misc.imsave(os.path.join('output','predict_%04d.png' %ii),np.uint8(output*255.0))
					scipy.misc.imsave(os.path.join('output','gt_%04d.png' %ii),np.uint8(gt))

				elif view==1:
					batch_files = range(test_input_horizontal.shape[-1])[ii*FLAGS.batch_size:(ii+1)*FLAGS.batch_size]	
					batches =[get_image_test(test_input_horizontal[0,batch],test_gt_horizontal[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					input_ = batches[:,:,:,:2]
					gt = batches[:,:,:,-1]
					start_time = time.time()
					output = LFSR.sess.run([LFSR.output_hor],feed_dict={LFSR.train_input_horizontal:input_})
					end_time = time.time()-start_time
					print('Test processing %d/%d time:%f \n' %(ii,test_input_horizontal.shape[-1],end_time))
					output = np.squeeze(output)
					gt = np.squeeze(gt)

					ssim_val += ssim_exact(gt.astype(float)/255.0,output)
					psnr_val += psnr(gt,np.uint8(output*255))			
					scipy.misc.imsave(os.path.join('output','predict_%04d.png' %ii),np.uint8(output*255.0))
					scipy.misc.imsave(os.path.join('output','gt_%04d.png' %ii),np.uint8(gt))
				else:
					batch_files = range(test_input_4views.shape[-1])[ii*FLAGS.batch_size:(ii+1)*FLAGS.batch_size]	
					batches =[get_image_test(test_input_4views[0,batch],test_gt_4views[0,batch]) for batch in batch_files]
					batches = np.array(batches).astype(np.float32)
					input_ = batches[:,:,:,:4]
					gt = batches[:,:,:,-1]
					start_time = time.time()
					output = LFSR.sess.run([LFSR.output_views],feed_dict={LFSR.train_input_4views:input_})
					end_time = time.time()-start_time
					print('Test processing %d/%d time:%f \n' %(ii,test_input_4views.shape[-1],end_time))
					output = np.squeeze(output)
					gt = np.squeeze(gt)

					ssim_val += ssim_exact(gt.astype(float)/255.0,output)
					psnr_val += psnr(gt,np.uint8(output*255))			
					scipy.misc.imsave(os.path.join('output','predict_%04d.png' %ii),np.uint8(output*255.0))
					scipy.misc.imsave(os.path.join('output','gt_%04d.png' %ii),np.uint8(gt))

				"""
				#pixel =5
				for nn in range(FLAGS.batch_size):
					#scipy.misc.imsave(os.path.join('output','predict_%04d.png' %ii),output)
					#scipy.misc.imsave(os.path.join('output','gt_%04d.png' %ii),np.uint8(label))
					if FLAGS.batch_size ==1:
						tmp_output =output
						tmp_gt =gt
						#tmp_output = output[pixel:-pixel,pixel:-pixel]
						#tmp_gt = gt[pixel:-pixel,pixel:-pixel]
						ssim_val += ssim_exact(tmp_gt.astype(float)/255.0,tmp_output)
						psnr_val += psnr(tmp_gt,np.uint8(tmp_output*255))			
						scipy.misc.imsave(os.path.join('output','predict_%04d.png' %ii),tmp_output)
						scipy.misc.imsave(os.path.join('output','gt_%04d.png' %ii),np.uint8(tmp_gt))
					else:
						tmp_output = np.squeeze(output[nn,pixel:-pixel,pixel:-pixel])
						tmp_gt = np.squeeze(gt[nn,pixel:-pixel,pixel:-pixel])
						ssim_val += ssim_exact(tmp_gt.astype(float)/255.0,tmp_output)
						psnr_val += psnr(tmp_gt,np.uint8(tmp_output*255))			
				"""		
			mean_ssim = ssim_val/test_input_vertical.shape[-1]	
			mean_psnr = psnr_val/test_input_vertical.shape[-1]	
			print('mean psnr: %.6f mean ssim: %.6f \n' %(mean_psnr,mean_ssim))

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
