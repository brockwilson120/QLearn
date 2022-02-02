from colordetector import colorDetect
from newGame import SimonSays
from collections import deque
import torch
from NNmodel import Linear_QNet, QTrainer
import random
import time
from helper import plot


MEM_MAX = 10_000
BATCHSIZE = 1000
learningRate = 0.001

#AGENT
class agent():

	def __init__(self):
		self.gamma = 0 # discount rate
		self.epsilon = 0 
		self.n_games = 0
		self.model = Linear_QNet(4, 256, 4)
		self.trainer = QTrainer(model=self.model, learningRate=learningRate, gamma=self.gamma)
		self.memory = deque(maxlen=MEM_MAX)


	def play_state(self):
		return colorDetect().detect()

	def get_action(self, state):
		
		final_move = [0, 0, 0, 0]

		self.epsilon = 50 - self.n_games
		if random.randint(0, 200) < self.epsilon:
			final_move[random.randint(0,3)] = 1
		else:
			moveNext = torch.tensor(state, dtype=torch.float)
			prediction = self.model(moveNext)
			move = torch.argmax(prediction).item()
			final_move[move] = 1
			
		return final_move

	def short_memory(self, state, action, reward, next_state, lost):
		self.trainer.train_iteration(state, action, reward, next_state, lost)
		# SEND ALL THIS OFF TO Q TRAINER

	def long_memory(self, state, action, reward, next_state, lost):
		if BATCHSIZE < len(self.memory):
			mini_sample = random.sample(self.memory, BATCHSIZE)
		else:
			mini_sample = self.memory


	def remember(self, oldState, final_move, reward, newState, lost):
		self.memory.append((oldState, final_move, reward, newState, lost))


def startTrain():
	Agent = agent()
	game = SimonSays()
	score = 0
	record = 0
	reward = 0

	plotScores = []
	plotAvgScores = []

	while 1:

		oldState = Agent.play_state()

		final_move = Agent.get_action(oldState)

		score, lost, reward = game.read_key(final_move)

		newState = Agent.play_state()

		#NEED MEMORY
		Agent.short_memory(oldState, final_move, reward, newState, lost)

		#MAKE REMEMBER
		Agent.remember(oldState, final_move, reward, newState, lost)



		# score, lost = game.read_key(action)
		if lost:
			game.reset()

			Agent.n_games += 1

			if score > record:
				record = score
				Agent.model.save()

			plotScores.append(score)
			plotAvgScores.append(score / Agent.n_games)
			plot(plotScores, plotAvgScores)

			
			print('Games: ', Agent.n_games, 'Final score: ', score, 'Record: ', record,'\n')


if __name__ == '__main__':
	startTrain()
	

	'''

	by game: 50 Final score: 567 Record: 567
	'''
