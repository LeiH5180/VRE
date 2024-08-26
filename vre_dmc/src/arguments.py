import argparse
import numpy as np
import random


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--eval_mode_1', default='color_hard', type=str)
	parser.add_argument('--eval_mode_2', default='video_easy', type=str)
	parser.add_argument('--eval_mode_3', default='video_hard', type=str)
	parser.add_argument('--if_base', default='vre', type=str)
	parser.add_argument('--only_VRE', default=True, type=str)    # no KL_loss for actor
	parser.add_argument('--add_VRE', default=True, type=str)
	parser.add_argument('--double_aug', default=True, type=str)    # if conv and over
	parser.add_argument('--over_rand', default=False, type=str)    # if rand para for overlay


	parser.add_argument('--KL_encoder', default=False, type=str)
	parser.add_argument('--kl_lim', default=False, type=str)
	parser.add_argument('--kl_tar', default=0.25, type=int)
	parser.add_argument('--auto_para', default=False, type=str)
	parser.add_argument('--auto_Q', default=False, type=str)
	parser.add_argument('--Vre_para', default=0.1, type=int)
	parser.add_argument('--critic_para', default=1, type=int)
	
	
	# agent
	parser.add_argument('--algorithm', default='vre', type=str)
	parser.add_argument('--train_steps', default='500000', type=str)    
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1, type=int)    # 1000
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)
	parser.add_argument('--encoder_update_freq', default=1, type=int)
	parser.add_argument('--kl_beta', default=0.1, type=float)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# encoder-----------------------------------------------------
	parser.add_argument('--encoder_lr', default=0.0001, type=float)
	parser.add_argument('--encoder_beta', default=0.9, type=float)
	parser.add_argument('--encoder_multi', default=0, type=float)    # set 0
	# ------------------------------------------------------------

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--a_alpha', default=0.5, type=float)
	parser.add_argument('--b_beta', default=0.5, type=float)
	parser.add_argument('--g_gamma', default=0.5, type=float)

	# eval
	parser.add_argument('--save_freq', default='500k', type=str)
	parser.add_argument('--eval_freq', default='10000', type=str)
	parser.add_argument('--eval_episodes', default=0, type=int)
	parser.add_argument('--distracting_cs_intensity', default=0.5, type=float)

	# misc
	parser.add_argument('--seed', default=3, type=int)
	parser.add_argument('--de_num', default=0, type=int)
	parser.add_argument('--log_dir', default='logs_VRE_walkerrun_common_lr0.0001_upf1_obscat_autopara0.02_Q', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--save_buffer', default=False, type=str)
	parser.add_argument('--buffer_save_freq', default='70k', type=str)

	args = parser.parse_args()

	assert args.algorithm in {'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', 'sgqn', 'dtk', 'vre'}, f'specified algorithm "{args.algorithm}" is not supported'

	assert args.eval_mode_1 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_1}" is not supported'
	assert args.eval_mode_2 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_2}" is not supported'
	assert args.eval_mode_3 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_2}" is not supported'

	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	if args.eval_mode_1 == 'none':
		args.eval_mode_1 = None
	if args.eval_mode_2 == 'none':
		args.eval_mode_2 = None
	if args.eval_mode_3 == 'none':
		args.eval_mode_3 = None

	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
	else:
		args.image_size = 84
		args.image_crop_size = 84
	
	return args
