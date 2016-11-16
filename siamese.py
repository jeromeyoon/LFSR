import os
import time
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
import pdb
import scipy.io as sio
from utils import *
from ops import *
from load_data import*
class Model(object):
	def __init__(self,sess,date,image_wid=64,image_hei=64,batch_size=32,dataset_name='default',checkpoint_dir=None):
		self.sess = sess
		self.batch_size = batch_size
		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.image_wid = image_wid
		self.image_hei = image_hei
		self.LF_wid = 768
		self.LF_hei = 768
		self.count = 0
		self.date =date
		self.queue = queue
		self.build_model()
	def build_model(self):
		if not self.queue:
			self.train_input_vertical = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,2],name ='train_input_vertical')
			self.train_input_horizontal = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,2],name ='train_input_horizontal')
			self.train_input_4views = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,4],name ='train_input_4views')
			self.train_gt = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_gt')
		else:
			print "using queue loading"
			self.train_input_ver = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,2],name ='train_input_vertical')
			self.train_input_hor = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,2],name ='train_input_horizontal')
			self.train_input_cen = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,4],name ='train_input_4views')
			self.train_gti_queue = tf.placeholder(tf.float32,[self.batch_size,self.image_hei,self.image_wid,1],name ='train_gt')
			q = tf.FIFOQUEUE(10000,[tf.floate32,tf,float32,tf.float32,tf.float32],[[self.image_hei,self.image_wid,2],[self.image_hei,self.image_wid,2],[self.image_hei,self.image_wid,4],[self.image_hei,self.image_wid,1]])
			enqueue_op = q.enqueue([self.train_input_ver,self.train_input_hor,self.train_input_cen,self.train_gt_queue])
			self.train_input_vertical,self.train_input_horizontal,self.train_input_4views,self.train_gt = q.dequeue(self.batch_size)
				

		self.ver_output = self.vertical_net(self.train_input_vertical)
		self.hor_output = self.horizontal_net(self.train_input_horizontal)
		self.views_output = self.views_net(self.train_input_4views)
		self.output_ver = self.shared_net(self.ver_output)
		self.output_hor = self.shared_net(self.hor_output,reuse=True )
		self.output_views = self.shared_net(self.views_output,reuse=True)
		
		self.loss_ver = tf.reduce_mean(tf.square(tf.sub(self.output_ver,self.train_gt)))# MSE
		self.loss_hor = tf.reduce_mean(tf.square(tf.sub(self.output_hor,self.train_gt)))# MSE
		self.loss_views = tf.reduce_mean(tf.square(tf.sub(self.output_views,self.train_gt)))# MSE
		self.saver = tf.train.Saver(max_to_keep=1)
	def train(self,config):
		global_step1 = tf.Variable(0,name='global_step_train1',trainable=False)	
		global_step2 = tf.Variable(0,name='global_step_train2',trainable=False)	
		global_step3 = tf.Variable(0,name='global_step_train3',trainable=False)	
		train_optim1 = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_ver,global_step=global_step1)
		train_optim2 = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_hor,global_step=global_step2)
		train_optim3 = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_views,global_step=global_step3)
		tf.initialize_all_variables().run()

		# Load training data

		train_input_vertical,train_input_horizontal,train_input_4views,train_gt_vertical,train_gt_horizontal,train_gt_4views = load_traindata() 
		val_input_vertial,val_input_horizontal,val_input_4views,val_gt_vertical,val_gt_horizontal,val_gt_4views  = load_valdata()
		batch_idxs_views = (train_input_4views.shape[-1])/config.batch_size
		val_batch_idxs_views = (val_input_4views.shape[-1])/config.batch_size

		#load trained network
		if self.loadnet(self.checkpoint_dir):
			print('Load pretrained network')
		else:
			print(' Load Fail!!')


		def load_and_enqueue(coord,file_list,idx=0,num_thread=1):


		if self.queue:
			coord = tf.train.Coordinator()
			num_thread =5
			for u in range(num_thread):
			






		if not self.queue:
			
			for epoch in xrange(config.epochs):
				rand_idx_ver = np.random.permutation(range(train_input_vertical.shape[-1]))
				rand_idx_hor = np.random.permutation(range(train_input_horizontal.shape[-1]))
				rand_idx_views = np.random.permutation(range(train_input_4views.shape[-1]))
				val_rand_idx_ver = np.random.permutation(range(val_input_vertical.shape[-1]))
				val_rand_idx_hor = np.random.permutation(range(val_input_horizontal.shape[-1]))
				val_rand_idx_views = np.random.permutation(range(val_input_4views.shape[-1]))

				sum_train_MSE = 0.0
				sum_val_MSE =0.0
				for idx in xrange(0,batch_idxs_views):
					if epoch ==0:
						f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'w')
						f_val = open(os.path.join("logs",self.date,'val.log'),'w')
					else:
						f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'aw')
						f_val = open(os.path.join("logs",self.date,'val.log'),'aw')

					randview =np.random.permutation(range(3))
					for view in randview:
						if view ==0:
							batch_files = rand_idx_ver[idx*config.batch_size:(idx+1)*config.batch_size]	
							batches =[get_image(train_input_vertical[0,batch],train_gt_vertical[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:2]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim1,self.loss_ver],feed_dict={self.train_input_vertical:input_,self.train_gt:gt})
							self.count +=1
							sum_train_MSE +=MSE

						elif view==1:
							batch_files = rand_idx_hor[idx*config.batch_size:(idx+1)*config.batch_size]	
							batches =[get_image(train_input_horizontal[0,batch],train_gt_horizontal[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:2]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim2,self.loss_hor],feed_dict={self.train_input_horizontal:input_,self.train_gt:gt})
							self.count +=1
							sum_train_MSE +=MSE
						else:
							batch_files = rand_idx_views[idx*config.batch_size:(idx+1)*config.batch_size]	
							batches =[get_image(train_input_4views[0,batch],train_gt_4views[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:4]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim3,self.loss_views],feed_dict={self.train_input_4views:input_,self.train_gt:gt})
						self.count +=1
						sum_train_MSE +=MSE

				print('Epoch train[%2d] MSE: %.4f \n' %(epoch,sum_train_MSE/(3*batch_idxs_views)))
				#Validation
				for val_idx in xrange(0,val_batch_idxs_views):		
					
					randview =np.random.permutation(range(3))
					for view in randview:
						if view ==0:
							batch_files = val_rand_idx_ver[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
							batches =[get_image(val_input_vertical[0,batch],val_gt_vertical[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:2]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							MSE = self.sess.run([self.loss_ver],feed_dict={self.train_input_vertical:input_,self.train_gt:gt})
							self.count +=1
							sum_train_MSE +=MSE

						elif view==1:
							batch_files = val_rand_idx_hor[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
							batches =[get_image(val_input_horizontal[0,batch],val_gt_horizontal[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:2]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							MSE = self.sess.run([self.loss_hor],feed_dict={self.train_input_horizontal:input_,self.train_gt:gt})
							self.count +=1
							sum_train_MSE +=MSE
						else:
							batch_files = val_rand_idx_views[val_idx*config.batch_size:(val_idx+1)*config.batch_size]	
							batches =[get_image(val_input_4views[0,batch],val_gt_4views[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:4]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							MSE = self.sess.run([self.loss_views],feed_dict={self.train_input_4views:input_,self.train_gt:gt})

						sum_val_MSE +=MSE[0]
				print('Epoch val[%2d] MSE: %.4f \n' %(epoch,sum_train_MSE/(3*val_batch_idxs_views)))
				if np.mod(epoch,100) ==0:
					f_train_epoch.write('epoch %06d mean_MSE %.6f \n' %(epoch, sum_train_MSE/(3*np.float(batch_idxs_views))))
					f_train_epoch.close()
					f_val.write('epoch %06d MSE %.6f \n' %(epoch, sum_val_MSE/(3*np.float(val_batch_idxs_views))))
					f_val.close()	
					self.save(config.checkpoint_dir,0)
		else:
			print("Training using Queue")
			for epoch in xrange(config.epochs):
				rand_idx_ver = np.random.permutation(range(train_input_vertical.shape[-1]))
				rand_idx_hor = np.random.permutation(range(train_input_horizontal.shape[-1]))
				rand_idx_views = np.random.permutation(range(train_input_4views.shape[-1]))
				val_rand_idx_ver = np.random.permutation(range(val_input_vertical.shape[-1]))
				val_rand_idx_hor = np.random.permutation(range(val_input_horizontal.shape[-1]))
				val_rand_idx_views = np.random.permutation(range(val_input_4views.shape[-1]))

				sum_train_MSE = 0.0
				sum_val_MSE =0.0
				for idx in xrange(0,batch_idxs_views):
					if epoch ==0:
						f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'w')
						f_val = open(os.path.join("logs",self.date,'val.log'),'w')
					else:
						f_train_epoch = open(os.path.join("logs",self.date,'train_epoch.log'),'aw')
						f_val = open(os.path.join("logs",self.date,'val.log'),'aw')

					randview =np.random.permutation(range(3))
					for view in randview:
						if view ==0:
							#batch_files = rand_idx_ver[idx*config.batch_size:(idx+1)*config.batch_size]	
							#batches =[get_image(train_input_vertical[0,batch],train_gt_vertical[0,batch],self.image_wid) for batch in batch_files]
							#batches = np.array(batches).astype(np.float32)
							#input_ = batches[:,:,:,:2]
							#gt = batches[:,:,:,-1]
							#gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim1,self.loss_ver])
							self.count +=1
							sum_train_MSE +=MSE

						elif view==1:
							batch_files = rand_idx_hor[idx*config.batch_size:(idx+1)*config.batch_size]	
							batches =[get_image(train_input_horizontal[0,batch],train_gt_horizontal[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:2]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim2,self.loss_hor],feed_dict={self.train_input_horizontal:input_,self.train_gt:gt})
							self.count +=1
							sum_train_MSE +=MSE
						else:
							batch_files = rand_idx_views[idx*config.batch_size:(idx+1)*config.batch_size]	
							batches =[get_image(train_input_4views[0,batch],train_gt_4views[0,batch],self.image_wid) for batch in batch_files]
							batches = np.array(batches).astype(np.float32)
							input_ = batches[:,:,:,:4]
							gt = batches[:,:,:,-1]
							gt = np.expand_dims(gt,axis=-1)
							_,MSE = self.sess.run([train_optim3,self.loss_views],feed_dict={self.train_input_4views:input_,self.train_gt:gt})
						self.count +=1
						sum_train_MSE +=MSE

				print('Epoch train[%2d] MSE: %.4f \n' %(epoch,sum_train_MSE/(3*batch_idxs_views)))

	def vertical_net(self,ver_input):
		h1 = tf.nn.relu(conv2d(ver_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='ver' ))
		return h1

	def horizontal_net(self,hor_input):
		h1 = tf.nn.relu(conv2d(hor_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='hor' ))
		return h1
	def views_net(self,views_input):
		h1 = tf.nn.relu(conv2d(views_input,64,k_h=9,k_w=9,d_h=1,d_w=1,padding='SAME',name='views' ))
		return h1

	
	def shared_net(self,input_,reuse=None):
		with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
			h1 = tf.nn.relu(conv2d(input_,32,k_h=5,k_w=5,d_h=1,d_w=1,padding='SAME',name='shread_1' ))
			h2 = conv2d(h1,1,k_h=5,k_w=5,d_h=1,d_w=1,padding='SAME',name='shared_2')
			return h2
	
	def loadnet(self,checkpoint_dir):
		model_dir = '%s' %(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
		ckpt = 	tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess,os.path.join(checkpoint_dir,ckpt_name))	
			return True
		else:
			return False
	def save(self, checkpoint_dir, step):
        	model_name = "DCGAN.model"
	        model_dir = "%s" % (self.dataset_name)
	        #model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

	        if not os.path.exists(checkpoint_dir):
        	    os.makedirs(checkpoint_dir)

	        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),global_step=step)


	def view_agent(self,views,batches):
		
		enc = np.zeros([self.batch_size,5])
		for idx,batch in enumerate(batches):
			enc[idx,views[batch]] =1.		
		return np.array(enc).astype(np.float32)

