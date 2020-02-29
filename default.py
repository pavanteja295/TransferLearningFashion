import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchvision.models as models
from utils.metric import accuracy, AverageMeter, Timer
from focalloss import FocalLoss
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
import os
from resnet import RESNET18


class Network_(nn.Module):
	def __init__(self, agent_config):
		'''
		:param agent_config (dict): lr=float,momentum=float,weight_decay=float,
									schedule=[int],  # The last number in the list is the end of epoch
									model_type=str,model_name=str,out_dim={task:dim},model_weights=str
									force_single_head=bool
									print_freq=int
									gpuid=[int]'''

		super(Network_, self).__init__()
		self.log = print
		self.config = agent_config

		self.train_loader, self.test_loader = self.config['loaders']['pretrain']

		self.model = self.create_model()
		self.criterion_fn = FocalLoss() if self.config['loss'] == 'fl' else CrossEntropyLoss() 

		if agent_config['gpuid'][0] >= 0:
			self.cuda()
			self.gpu = True
		else:
			self.gpu = False
		
		self.exp_name = agent_config['exp_name']
		self.init_optimizer()

		self.writer = SummaryWriter(log_dir="runs/" + self.exp_name)
		self.save_after = self.config['save_after']

	def init_optimizer(self):
		optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD','RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
			optimizer_arg['nesterov'] = self.config['nesterov']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'

		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=self.config['gamma'] )

	# def freeze_layers(self):
	# 	if self.config['freeze']:

	def create_model(self):
		if self.config['model_name'] == 'RESNET18':
			model = RESNET18()
		else:
			model = models.__dict__[self.config['model_name']](pretrained=self.config['pretrain_in'])

		# Freeze training for all "features" layers
		if self.config['freeze'] != 'None':
			for param in model.parameters():
				param.requires_grad = False

		n_inp = model.fc.in_features
		model.fc = nn.Linear(n_inp, len(self.train_loader.dataset.class_list))
		# Load pre-trained weights
		if self.config['model_weights'] is not None:
			print('=> Load model weights:', self.cfg['model_weights'])
			model_state = torch.load(self.cfg['model_weights'],
										map_location=lambda storage, loc: storage)  # Load to CPU.
			model.load_state_dict(model_state)
			print('=> Load Done')
		return model
	
	def criterion(self, preds, targets):
		loss = self.criterion_fn(preds, targets)
		return loss

	def cuda(self):
		torch.cuda.set_device(self.config['gpuid'][0])
		self.model = self.model.cuda()
		self.criterion_fn = self.criterion_fn.cuda()
		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
		return self

	def forward(self, x):
		return self.model.forward(x)

	def switch_finetune(self):
		print('Switched to new task FINETUNING')
		self.train_loader, self.test_loader = self.config['loaders']['finetune']


		# Freeze training for all "features" layers
		for param in self.model.parameters():
			param.requires_grad = False

		
		print('FINETUNING number of classes are ', len(self.train_loader.dataset.class_list))
		n_inp = self.model.fc.in_features
		self.model.fc = nn.Linear(n_inp, len(self.train_loader.dataset.class_list))

		self.config['lr'] = 0.001
		self.init_optimizer()
		self.cuda()


		# switch train and test loaders
		# switch model's layers or freeze them
		# change lr or criterion if required
		# change tensorboard suffixes
		return

	def accumulate_acc(self, output, target, meter):
		meter.update(accuracy(output, target), len(target))
		return meter

	def update_model(self, inputs, targets):
		out = self.forward(inputs)
		loss = self.criterion(out, targets)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out

	# def validation(self):
    #     # this might possibly change for other incremental scenario
    #     # This function doesn't distinguish tasks.
	# 	batch_timer = Timer()
	# 	acc = AverageMeter()
	# 	losses = AverageMeter()
	# 	acc_5 = AverageMeter()
	# 	acc_class =  [AverageMeter() for i in range(len(self.train_loader.dataset.class_list))]  #[AverageMeter()] *  len(self.train_loader.dataset.class_list)
	# 	acc_class_5 = [AverageMeter() for i in range(len(self.train_loader.dataset.class_list))]  
	# 	batch_timer.tic()
	# 	orig_mode = self.training	
	# 	self.eval()
	# 	for i, (input, target) in enumerate(self.test_loader):

	# 		if self.gpu:
	# 			print(self.gpu)
	# 			with torch.no_grad():
	# 				input = input.cuda()
	# 				target = target.cuda()
	# 		output = self.forward(input)
	# 		loss = self.criterion(output, target)
	# 		losses.update(loss, input.size(0))        
	# 		# Summarize the performance of all tasks, or 1 task, depends on dataloader.
	# 		# Calculated by total number of data.
			
	# 		#t_acc, acc_class = accuracy(output, target, topk=(1,), avg_meters=acc_class) #self.accumulate_acc(output, target, acc)
	# 		#t_acc_5, acc_class_5 = accuracy(output, target, topk=(5,), avg_meters=acc_class_5)
	# 		# import pdb; pdb.set_trace()
	# 		acc.update(t_acc, len(target))
	# 		acc_5.update(t_acc_5, len(target))

	# 	class_list = self.train_loader.dataset.class_list.inverse
	# 	acc_cl_1 = {}
	# 	acc_cl_5 = {}
		
	# 	# for idx, cl_ in class_list.items():
	# 	# 	acc_cl_1[cl_] = [acc_class[idx].avg,  acc_class[idx].sum, acc_class[idx].count] 
	# 	# 	acc_cl_5[cl_] = [acc_class_5[idx].avg,  acc_class_5[idx].sum,  acc_class_5[idx].count]
	# 	# 	self.log(' * Val Acc {acc.avg:.3f} for class {cls}, {acc.sum} / {acc.count} '
	# 	# 		.format(acc=acc_class[idx], cls=cl_))
		

	# 	self.train(orig_mode)

	# 	self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
	# 			.format(acc=acc,time=batch_timer.toc()))
	# 	return acc, losses

	def predict(self, inputs):
		self.model.eval()
		out = self.forward(inputs)
		return out


	def validation(self):
		# this might possibly change for other incremental scenario
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()

		batch_timer.tic()

		orig_mode = self.training
		self.eval()
		for i, (input, target) in enumerate(self.test_loader):

			if self.gpu:
				with torch.no_grad():
					input = input.cuda()
					target = target.cuda()

			output = self.predict(input)
			loss = self.criterion(output)
			losses.update(loss, input.size(0))        
			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			acc = accumulate_acc(output, target, acc)

		self.train(orig_mode)

		self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
				.format(acc=acc,time=batch_timer.toc()))
		return acc, losses

	def save_model(self, filename):
		dir_ = os.path.join('models', self.exp_name)
		if not os.path.exists(dir_):
			os.makedirs(dir_)
		model_state = self.model.state_dict()
		for key in model_state.keys():  # Always save it to cpu
			model_state[key] = model_state[key].cpu()
		print('=> Saving model to:', filename)
		torch.save(model_state, os.path.join(dir_, filename + '.pth'))
		print('=> Save Done')

	def train_(self, epochs, finetune=False):
		str_ = 'pretrain'
		if finetune:
			self.switch_finetune()
			str_ = 'finetune'
		# else:
		# 	self.switch_to_pretrain()


		for epoch in range(epochs):

			data_timer = Timer()
			batch_timer = Timer()
			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()
			acc = AverageMeter()
			self.model.train()
			self.scheduler.step(epoch)
			for param_group in self.optimizer.param_groups:
				self.log('LR:',param_group['lr'])

			self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
			self.log('{0} Epoch:{1}'.format(str_, epoch))
			
			data_timer.tic()
			batch_timer.tic()
			
			for i, (input, target) in enumerate(self.train_loader):
				self.model.train()
				data_time.update(data_timer.toc())  # measure data loading time

				if self.gpu:
					input = input.cuda()
					target = target.cuda()

				loss, output = self.update_model(input, target)
				input = input.detach()
				target = target.detach()

				# measure accuracy and record loss
				acc = self.accumulate_acc(output, target, acc)
				losses.update(loss, input.size(0))
				batch_time.update(batch_timer.toc())  # measure elapsed time
				data_timer.toc()
				self.n_iter = (epoch) * len(self.train_loader)  + i
				self.writer.add_scalar(str_ + '/Loss_train', losses.avg, self.n_iter)
				self.writer.add_scalar(str_ + '/Acc_train' , acc.avg, self.n_iter)
                # if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
				self.log('[{0}/{1}]\t'
						'{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
						'{data_time.val:.4f} ({data_time.avg:.4f})\t'
						'{loss.val:.3f} ({loss.avg:.3f})\t'
						'{acc.val:.2f} ({acc.avg:.2f})'.format(
					i, len(self.train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, acc=acc))
				
		

			acc_v, loss_v = self.validation()
			self.writer.add_scalar(str_ + '/Loss_test', loss_v.avg, self.n_iter)
			self.writer.add_scalar(str_ + '/Acc_test' , acc_v.avg, self.n_iter)

			if epoch % self.save_after == 0 and epoch!=0:
				self.save_model(str_ + str(epoch) )



	



def accumulate_acc(output, target,meter):
# Single-headed model
    meter.update(accuracy(output, target), len(target))
    return meter
