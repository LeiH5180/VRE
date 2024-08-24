import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from torch.utils.tensorboard import SummaryWriter
from pyvirtualdisplay import Display

def evaluate(env, agent, video, num_episodes, L, step, test_buffer=None, test_env=False): 
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False 
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			video.record(env)
			if test_buffer is not None:
				test_buffer.add(obs, action, reward, next_obs)
			episode_reward += reward
			obs = next_obs

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')   
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,   
		task_name=args.task_name,    
		seed=args.seed,
		episode_length=args.episode_length,   
		action_repeat=args.action_repeat,    
		image_size=args.image_size,   
		mode='train'
	)
	test_env_1 = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode_1,
		intensity=args.distracting_cs_intensity
	) if args.eval_mode_1 is not None else None
	test_env_2 = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode_2,
		intensity=args.distracting_cs_intensity
	) if args.eval_mode_3 is not None else None
	test_env_3 = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode_3,
		intensity=args.distracting_cs_intensity
	) if args.eval_mode_3 is not None else None
	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name+'_'+args.algorithm+'_'+'1'+str(args.seed)+'_'+args.eval_mode_1+'_'+args.eval_mode_2+args.if_base+str(time.time()))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	tb_dir = work_dir + '//tb'
	writter = SummaryWriter(tb_dir)
	assert not os.path.exists(os.path.join(tb_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)   
	utils.write_info(args, os.path.join(work_dir, 'info.log'))  

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps // 2,
		batch_size=args.batch_size,
		encoder_multi=args.encoder_multi
	)
	test_buffer_1 = None
	test_buffer_2 = None
	test_buffer_3 = None
	if args.save_buffer:
		test_buffer_1 = utils.Test_Buffer(
			obs_shape=env.observation_space.shape,
			action_shape=env.action_space.shape,
			capacity=args.train_steps // 6,
		)
		test_buffer_2 = utils.Test_Buffer(
			obs_shape=env.observation_space.shape,
			action_shape=env.action_space.shape,
			capacity=args.train_steps // 6,
		)
		test_buffer_3 = utils.Test_Buffer(
			obs_shape=env.observation_space.shape,
			action_shape=env.action_space.shape,
			capacity=args.train_steps // 6,
		)	
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)    #(9, 84, 84)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	kl_loss = None
	eval_array = np.zeros(args.train_steps//args.eval_freq)    # 记录eval的reward
	cof = 1
	for step in range(start_step, args.train_steps+1):
		if done or step == args.train_steps:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				eval_reward = evaluate(env, agent, video, args.eval_episodes, L, step)
				writter.add_scalar('eval_reward', eval_reward, step)
				eval_array[step//args.eval_freq] = eval_reward
				if step >= 200000:
					pos = step // args.eval_freq
					cof = (((eval_array[pos]+eval_array[pos-1])-(eval_array[pos-2]+eval_array[pos-3]))
								/(eval_array[pos]+eval_array[pos-1]))
					L.log('eval/cof', cof, step)
				if test_env_1 is not None:
					test_reward_1 = evaluate(test_env_1, agent, video, args.eval_episodes, L, step, test_buffer_1, test_env=True)
					writter.add_scalar('test_reward'+args.eval_mode_1, test_reward_1, step)
				L.dump(step)
				if test_env_2 is not None:
					test_reward_2 = evaluate(test_env_2, agent, video, args.eval_episodes, L, step, test_buffer_2, test_env=True)
					writter.add_scalar('test_reward'+args.eval_mode_2, test_reward_2, step)				
				L.dump(step)
				if test_env_3 is not None:
					test_reward_3 = evaluate(test_env_3, agent, video, args.eval_episodes, L, step, test_buffer_3, test_env=True)
					writter.add_scalar('test_reward'+args.eval_mode_3, test_reward_3, step)				
				L.dump(step)
				if kl_loss is not None:
					writter.add_scalar('kl_loss', kl_loss, step)
					L.log('eval/kl_loss', kl_loss, step)
					L.dump(step)
			# Save agent periodically
			if step > start_step and step % args.save_freq == 0: 
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))
				if args.save_buffer and step % args.buffer_save_freq == 0:
					torch.save(replay_buffer, os.path.join(work_dir, f'replay_buffer_{step}.pt'))
					torch.save(test_buffer_1, os.path.join(work_dir, f'test_buffer_1_{step}.pt'))
					torch.save(test_buffer_2, os.path.join(work_dir, f'test_buffer_2_{step}.pt'))
					torch.save(test_buffer_3, os.path.join(work_dir, f'test_buffer_3_{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			writter.add_scalar('train_episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)


		if step < args.init_steps:
			action = env.action_space.sample()  
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				if args.algorithm == 'VRE':
					# if step == 248:
					# 	time.sleep(0.01)
					# 	pass
					kl_loss = agent.update(replay_buffer, L, step, cof)
					if kl_loss is not None:
						writter.add_scalar('train_kl_loss', kl_loss, step, cof)
				else:
					agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	display = Display(visible=0, size=(200, 200))
	display.start()
	main(args)
	display.stop()
