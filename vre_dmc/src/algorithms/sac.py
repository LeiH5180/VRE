import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
import augmentations


class SAC(object):
	def __init__(self, obs_shape, action_shape, args):
		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq
		self.encoder_update_freq = args.encoder_update_freq
		self.de_num = args.de_num
		self.add_VRE = args.add_VRE
		self.VRE_para = args.VRE_para
		self.algorithm = args.algorithm

		shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda(self.de_num)    # shared_11,filters_32
		head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda(self.de_num)
		actor_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		self.critic_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		if self.add_VRE or self.algorithm =='dtk':
			self.actor = m.Actor(self.critic_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda(self.de_num)
			self.actor_optimizer = torch.optim.Adam(
				self.actor.mlp.parameters(), lr=args.actor_lr, betas=(args.encoder_beta, 0.999)
			)				
		else:
			self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda(self.de_num)
			self.actor_optimizer = torch.optim.Adam(
				self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
			)
		self.critic = m.Critic(self.critic_encoder, action_shape, args.hidden_dim).cuda(self.de_num)
		self.critic_target = deepcopy(self.critic)

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda(self.de_num)
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)


		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)

		# -------------------------------------------------------------------
		self.critic_encoder_optimizer = torch.optim.Adam(
			self.critic.encoder.parameters(), lr=args.encoder_lr, betas=(args.encoder_beta, 0.999)
		)	
		# -------------------------------------------------------------------

		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def eval(self):
		self.train(False)

	@property
	def alpha(self):
		return self.log_alpha.exp()
		
	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).cuda(self.de_num)
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
		return mu.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=True)
		return pi.cpu().data.numpy().flatten()

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1,
								 target_Q) + F.mse_loss(current_Q2, target_Q)
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


	def update_actor_and_alpha(self, obs, L=None, step=None, kl_loss = None, update_alpha=True):
		_, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

		if kl_loss is not None:
			L.log('train/kl_loss', kl_loss, step)
			actor_loss += kl_loss

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

	def soft_update_critic_target(self):
		utils.soft_update_params(
			self.critic.Q1, self.critic_target.Q1, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.Q2, self.critic_target.Q2, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.encoder, self.critic_target.encoder,
			self.encoder_tau
		)

	def update_encoder(self, obs_1, action_1, obs_2, action_2, L=None, step=None):

		obs_all = torch.cat((obs_1, obs_2), dim=0)
		obs_aug = augmentations.random_overlay(obs_all.clone())
		action_all = torch.cat((action_1, action_2), dim=0)

		current_Q1_ori, current_Q2_ori = self.critic(obs_all, action_all)
		Q_ori = torch.min(current_Q1_ori, current_Q2_ori)
		norm2_Q_ori = torch.norm(Q_ori, p=2)
		current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action_all)
		Q_aug = torch.min(current_Q1_aug, current_Q2_aug)

		emb_ori = self.critic.encoder(obs_all)
		norm2_emb_ori = torch.norm(emb_ori, p=2)
		emb_aug = self.critic.encoder(obs_aug)

		dis_Q = torch.norm((Q_ori.detach()-Q_aug),p=2)
		dis_emb = torch.norm((emb_ori.detach()-emb_aug),p=2)
		loss_critic_encoder_5 = torch.norm(dis_Q/norm2_Q_ori.detach()-dis_emb/norm2_emb_ori.detach(),p=2)
		loss_critic_encoder = loss_critic_encoder_5 * self.VRE_para

		self.critic_encoder_optimizer.zero_grad()
		loss_critic_encoder.backward()
		self.critic_encoder_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample()
		obs = augmentations.random_shift(obs)
		# obs = augmentations.random_conv(obs)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.encoder_update_freq == 0 and self.add_VRE:
			obs_2, action_2 = replay_buffer.sample_encoder_sac()
			self.update_encoder(obs, action, obs_2, action_2, L=None, step=None)		

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

