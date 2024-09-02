import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class VRE(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.double_aug = args.double_aug
		self.a_alpha = args.a_alpha
		self.b_beta = args.b_beta
		self.g_gamma = args.g_gamma
		if self.double_aug:
			self.b_beta /= 2
			self.g_gamma /= 2
		self.over_rand = args.over_rand
		if self.over_rand:
			self.overlay = augmentations.random_overlay_rand
		else:
			self.overlay = augmentations.random_overlay
		self.encoder_update_freq = args.encoder_update_freq
		self.only_VRE = args.only_VRE

		self.eval_freq = args.eval_freq
		self.kl_beta = args.kl_beta
		self.kl_lim = args.kl_lim
		self.kl_tar = args.kl_tar
		self.VRE_para = args.VRE_para
		self.critic_para = args.critic_para
		self.auto_para = args.auto_para
		self.auto_Q = args.auto_Q
		self.add_VRE = args.add_VRE

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, obs_2=None, action_2=None, update_encoder=False):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)    
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)
			# -------------------------------------------------------------------
			obs_ori = obs
			action_ori = action
			# -------------------------------------------------------------------

		if not self.double_aug:
			obs_1_aug_1 = self.overlay(obs.clone())
			obs_1_aug_2 = None
			obs = utils.cat(obs, obs_1_aug_1)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)    
			critic_loss = (self.a_alpha + self.b_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))    
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_1_aug_1 = self.overlay(obs.clone())
			# plt.imshow(np.moveaxis(np.array(obs_1_aug_1[0][:3].cpu()),0,-1))    # show the aug figure
			# plt.savefig('aug_1.png')
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_1_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_1_aug_2 = augmentations.random_conv(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_1_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))		
			critic_loss *= self.critic_para


		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
	
		# --------------------------------------------------------------------
		obs_aug_1 = obs_1_aug_1 
		obs_aug_2 = obs_1_aug_2

		if update_encoder:
			if not self.double_aug:
				obs_aug_1, obs_aug_2 = self.update_encoder(obs_ori, action_ori, obs_2, action_2, obs_1_aug_1)
			else:
				obs_aug_1, obs_aug_2 = self.update_encoder(obs_ori, action_ori, obs_2, action_2, obs_1_aug_1, obs_1_aug_2)

		return obs_aug_1, obs_aug_2
		# --------------------------------------------------------------------

	def update_encoder(self, obs_1, action_1, obs_2, action_2, obs_1_aug_1 = None, obs_1_aug_2 = None):
		if not self.double_aug:
			if len(obs_2) != 0:
				obs_2_aug_1 = self.overlay(obs_2.clone())
				obs_aug_1 = torch.cat((obs_1_aug_1, obs_2_aug_1), dim=0)
			else:
				obs_aug_1 = obs_1_aug_1
				
			obs_aug_2 = None
			obs_all = torch.cat((obs_1, obs_2), dim=0)
			action_all = torch.cat((action_1, action_2), dim=0)

			current_Q1_ori, current_Q2_ori = self.critic(obs_all, action_all)
			Q_ori = torch.min(current_Q1_ori, current_Q2_ori)
			norm2_Q_ori = torch.norm(Q_ori, p=2)
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action_all)
			Q_aug_1 = torch.min(current_Q1_aug_1, current_Q2_aug_1)

			emb_ori = self.critic.encoder(obs_all)
			norm2_emb_ori = torch.norm(emb_ori, p=2)
			emb_aug_1 = self.critic.encoder(obs_aug_1)

			dis_Q_1 = torch.norm((Q_ori.detach()-Q_aug_1),p=2)
			dis_emb_1 = torch.norm((emb_ori.detach()-emb_aug_1),p=2)
			loss_critic_encoder_5 = torch.norm((dis_Q_1/norm2_Q_ori.detach())-dis_emb_1/norm2_emb_ori.detach(),p=2)
			loss_critic_encoder = loss_critic_encoder_5 * self.VRE_para								
			# ================================================================================

		else:
			if len(obs_2) != 0:
				obs_2_aug_1 = self.overlay(obs_2.clone())
				obs_2_aug_2 = augmentations.random_conv(obs_2.clone())
				obs_aug_1 = torch.cat((obs_1_aug_1, obs_2_aug_1), dim=0)
				obs_aug_2 = torch.cat((obs_1_aug_2, obs_2_aug_2), dim=0)
			else:
				obs_aug_1 = obs_1_aug_1
				obs_aug_2 = obs_1_aug_2

			obs_all = torch.cat((obs_1, obs_2), dim=0)
			action_all = torch.cat((action_1, action_2), dim=0)

			current_Q1_ori, current_Q2_ori = self.critic(obs_all, action_all)
			Q_ori = torch.min(current_Q1_ori, current_Q2_ori)
			norm2_Q_ori = torch.norm(Q_ori,p=2)
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action_all)
			Q_aug_1 = torch.min(current_Q1_aug_1, current_Q2_aug_1)
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action_all)
			Q_aug_2 = torch.min(current_Q1_aug_2, current_Q2_aug_2)
			emb_ori = self.critic.encoder(obs_all)
			norm2_emb_ori = torch.norm(emb_ori,p=2).detach()
			emb_aug_1 = self.critic.encoder(obs_aug_1)
			emb_aug_2 = self.critic.encoder(obs_aug_2)

			dis_Q_1 = torch.norm((Q_ori.detach()-Q_aug_1),p=2)
			dis_Q_2 = torch.norm((Q_ori.detach()-Q_aug_2),p=2)
			dis_emb_1 = torch.norm((emb_ori.detach()-emb_aug_1),p=2)
			dis_emb_2 = torch.norm((emb_ori.detach()-emb_aug_2),p=2)
			loss_critic_encoder_5 = (torch.norm((dis_Q_1/norm2_Q_ori.detach())-dis_emb_1/norm2_emb_ori.detach(),p=2)
							+ torch.norm((dis_Q_2/norm2_Q_ori.detach())-dis_emb_2/norm2_emb_ori.detach(),p=2))
			loss_critic_encoder = loss_critic_encoder_5 * self.VRE_para
			# =======================================================================================	
		
		self.critic_encoder_optimizer.zero_grad()
		loss_critic_encoder.backward()
		self.critic_encoder_optimizer.step()			

		return obs_aug_1, obs_aug_2


	def update(self, replay_buffer, L, step, cof = 1):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()
		update_encoder = False
		obs_2 = None
		action_2 = None
		kl_loss = None

		if self.auto_para:
			if 0 <= cof <= 0.02 and self.VRE_para < 1:
				self.VRE_para += 0.1
			else:
				if cof < -0.02 and self.VRE_para >= 0.05:    # cof<-0.02
					self.VRE_para /= 2

		if self.auto_Q and step > 250000:
			if 0 <= cof <= 0.02 and self.critic_para > 0.1:
				self.critic_para *= 0.8
			else:
				if cof < -0.02 and self.critic_para > 1:    # cof<-0.02
					self.critic_para /= 0.8
		
		if self.add_VRE and step % self.encoder_update_freq == 0:
			obs_2, action_2 = replay_buffer.sample_encoder()
			update_encoder = True

		obs_aug_1, obs_aug_2 = self.update_critic(obs, action, reward, next_obs, not_done, L, step, obs_2, action_2, update_encoder)

		if (step+1) % self.eval_freq == 0:

			mu_ori, _, _, log_std_ori = self.actor(obs,compute_pi=False, compute_log_pi=False)
			mu_1, _, _, log_std_1 = self.actor(obs_aug_1[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
			kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
			kl_loss = (kl_divergence_ori_aug_1).mean()

			if self.double_aug:
				mu_2, _, _, log_std_2 = self.actor(obs_aug_2[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
				kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)
				kl_loss = kl_loss + (kl_divergence_ori_aug_2).mean()

		if step % self.actor_update_freq == 0:
			if self.double_aug:
				# obs_aug = obs
				obs_aug = torch.cat((obs, obs_aug_1[0:obs.size(0)], obs_aug_2[0:obs.size(0)]), dim=0)
			else:
				# obs_aug = obs
				obs_aug = torch.cat((obs, obs_aug_1[0:obs.size(0)]), dim=0)

			if self.only_VRE:
				self.update_actor_and_alpha(obs_aug, L, step)
			else:    # add KL_loss to actor
				mu_ori, _, _, log_std_ori = self.actor(obs,compute_pi=False, compute_log_pi=False)
				mu_1, _, _, log_std_1 = self.actor(obs_aug_1[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
				kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
				kl_loss = (kl_divergence_ori_aug_1).mean()

				if self.double_aug:
					mu_2, _, _, log_std_2 = self.actor(obs_aug_2[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
					kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)
					kl_loss = kl_loss + (kl_divergence_ori_aug_2).mean()

				betta = self.kl_beta
				if self.kl_lim:
					if kl_loss < self.kl_tar:
						betta = 0    # keep kl
					
				self.update_actor_and_alpha(obs_aug, L, step, kl_loss=kl_loss*betta)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		return kl_loss

	def kl_divergence(self, mu_p, p_log_std, mu_q, q_log_std):
		p_std = torch.exp(p_log_std)
		q_std = torch.exp(q_log_std)

		kl = torch.log(q_std / p_std) + (p_std**2 + (mu_p - mu_q)**2) / (2 * q_std**2) - 0.5
		kl = torch.mean(kl, dim=-1)
		return kl
