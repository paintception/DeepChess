from copy import deepcopy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout

import time
import chess
import random

import numpy as np 

class GameHandler(object):

	# Example code of a competition between ANNs trained on different 
	# Classification Datasets: in order to reproduce the ones obtained
	# on Dataset 4 just change the architecture of the ANN according to
	# what is presented in my MSc Thesis 

	def __init__(self):
		self.StartingPositionsPath = 'PositionsSet.txt'
		self.MlpClassificationWeights = 'Bobbyweights.h5'
		self.MlpDimension = 768
		self.NumberSimulationGames = 2
		self.MlpWins = 0
		self.CnnWins = 0
		self.Draws = 0
		self.width = 8
		self.height = 8
		self.channels = 16
		self.ImportantSquareSet = chess.SquareSet(
			chess.BB_D4 | chess.BB_D5 |
			chess.BB_C4 | chess.BB_C5 |
			chess.BB_E4 | chess.BB_E5 |
			chess.BB_F2 | chess.BB_F7 |
			chess.BB_H2 | chess.BB_H7
			)

		self.SquareSet = chess.SquareSet(
			chess.BB_A1 | chess.BB_A2 | chess.BB_A3 | chess.BB_A4 | chess.BB_A5 |
			chess.BB_A6 | chess.BB_A7 | chess.BB_A8 | 
			chess.BB_B1 | chess.BB_B2 | chess.BB_B3 | chess.BB_B4 | chess.BB_B5 |
			chess.BB_B6 | chess.BB_B7 | chess.BB_B8 | 
			chess.BB_C1 | chess.BB_C2 | chess.BB_C3 | chess.BB_C4 | chess.BB_C5 |
			chess.BB_C6 | chess.BB_C7 | chess.BB_C8 | 
			chess.BB_D1 | chess.BB_D2 | chess.BB_D3 | chess.BB_D4 | chess.BB_D5 |
			chess.BB_D6 | chess.BB_D7 | chess.BB_D8 | 
			chess.BB_A1 | chess.BB_E2 | chess.BB_E3 | chess.BB_E4 | chess.BB_E5 |
			chess.BB_E6 | chess.BB_E7 | chess.BB_E8 | 
			chess.BB_F1 | chess.BB_F2 | chess.BB_F3 | chess.BB_F4 | chess.BB_F5 |
			chess.BB_F6 | chess.BB_F7 | chess.BB_F8 | 
			chess.BB_G1 | chess.BB_G2 | chess.BB_G3 | chess.BB_G4 | chess.BB_G5 |
			chess.BB_G6 | chess.BB_G7 | chess.BB_G8 | 
			chess.BB_H1 | chess.BB_H2 | chess.BB_H3 | chess.BB_H4 | chess.BB_H5 |
			chess.BB_H6 | chess.BB_H7 | chess.BB_H8
			)

	def evaluatePositionMlp(self, board, model, tmp_move, boardToPlay):

		pos = np.expand_dims(board, axis=0)
		out = model.predict(pos)

		return np.argmax(out)

	def evaluatePositionCnn(self, board, model, tmp_move, boardToPlay):

		pos = np.expand_dims(board, axis=0)
		out = model.predict(pos)

		return np.argmax(out)

	def loadCnnClassificationModel(self):

		CnnModel = Sequential()
		CnnModel.add(Convolution2D(20,5,5, border_mode="same", input_shape=(self.width, self.height, self.channels)))
		CnnModel.add(Activation("elu"))
		CnnModel.add(Convolution2D(50,3,3, border_mode="same", input_shape=(self.width, self.height, self.channels)))
		CnnModel.add(Activation("elu"))
		CnnModel.add(Dropout(0.25))
		CnnModel.add(Flatten())
		CnnModel.add(Dense(250, activation="elu"))
		CnnModel.add(Dense(12))
		CnnModel.add(Activation("softmax"))
		#CnnModel.load_weights('/home/matthia/Desktop/ThesisStuff/CnnWeights/Weights/CNNWeights.h5')

		return CnnModel

	def loadMlpClassificationModel(self):

		MlpModel = Sequential()
		MlpModel.add(Dense(2048, input_dim=self.MlpDimension, init='normal', activation='elu'))
		MlpModel.add(Dropout(0.2))
		MlpModel.add(Dense(2048, input_dim=self.MlpDimension, init='normal', activation='elu'))
		MlpModel.add(Dropout(0.2))
		MlpModel.add(Dense(1050, input_dim=self.MlpDimension, init='normal', activation='elu'))
		MlpModel.add(Dropout(0.2))
		MlpModel.add(Dense(8, init='normal', activation='elu'))
		MlpModel.add(Activation("softmax"))
		#MlpModel.load_weights("/home/matthia/Desktop/Bobby/8Classes/Bobbyweights.h5")

 		return MlpModel

 	def splitter(self, inputStr, black):

 		inputStr = format(inputStr, "064b")
		tmp = [inputStr[i:i+8] for i in range(0, len(inputStr), 8)]

		for i in xrange(0, len(tmp)):
			tmp2 = list(tmp[i])
			tmp2 = [int(x) * black for x in tmp2]
			tmp[i] = tmp2

		return tmp

 	def shapeBoardMlp(self, board):

		P = self.splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
		R = self.splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)			
		N = self.splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
		B = self.splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
		Q = self.splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)			
		K = self.splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

		p = self.splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
		r = self.splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
		n = self.splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
		b = self.splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
		q = self.splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
		k = self.splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

		l = P+R+N+B+Q+K+p+r+n+b+q+k

		BitMappedBoard = [item for sublist in l for item in sublist]

		return BitMappedBoard

	def shapeBoardCnn(self, board):

		CheckedInfo = self.is_checked(board)

		SquareAttackers = []
		PinnedSquares = []

		ImportantAttackers = []

		for square in self.SquareSet:
			if board.is_attacked_by(chess.WHITE, square):
				SquareAttackers.append(1)
			elif board.is_attacked_by(chess.BLACK, square):
				SquareAttackers.append(-1)
			else:
				SquareAttackers.append(0)

			if board.is_pinned(chess.WHITE, square):
				PinnedSquares.append(1)
			elif board.is_pinned(chess.BLACK, square):
				PinnedSquares.append(-1)
			else:
				PinnedSquares.append(0)

		if board.turn is True:
			SquareAttackers.append(1)
			PinnedSquares.append(1)

		elif board.turn is False:
			SquareAttackers.append(-1)
			PinnedSquares.append(-1)

		for ImportantSquare in self.ImportantSquareSet:
			WhiteAttackers = board.attackers(chess.WHITE, ImportantSquare)
			BlackAttackers = board.attackers(chess.BLACK, ImportantSquare)

			if len(WhiteAttackers) > len(BlackAttackers):
				ImportantAttackersFeatures = [1] * 64
			elif len(WhiteAttackers) < len(BlackAttackers):
				ImportantAttackersFeatures = [-1] * 64
			else:
				ImportantAttackersFeatures = [0] * 64

		simpleBoard = self.shapeBoardMlp(board)
	
		ConvfeaturedBoard = simpleBoard + CheckedInfo+SquareAttackers+PinnedSquares+ImportantAttackersFeatures

		return ConvfeaturedBoard

	def is_checked(self, board):

		if board.is_check() and board.turn is True:
			CheckedInfo = [-1] * 64

		elif board.is_check() and board.turn is False:
			CheckedInfo = [1] * 64

		elif not board.is_check():
			CheckedInfo = [0] * 64

		return CheckedInfo

	def loadStartingPositions(self, position):
		return chess.Board(fen=position)
	
	def chooseWhite(self):
		return random.randint(0,1)

	def createSetMoves(self, board):
		return board.legal_moves

	def updateGameStatsWhite(self, result):

		if result == '1-0':
			self.MlpWins = self.MlpWins + 1
		elif result == '0-1':
			self.CnnWins = self.CnnWins + 1
		elif result == '1/2-1/2':
			self.Draws = self.Draws + 1
		else:
			pass

	def updateGameStatsBlack(self, result):

		if result == '1-0':
			self.CnnWins = self.CnnWins + 1
		elif result == '0-1':
			self.MlpWins = self.MlpWins + 1
		elif result == '1/2-1/2':
			self.Draws = self.Draws + 1
		else:
			pass

	def makeCandidateMovesMlp(self, boardToPlay, setMoves, MlpModel):
		
		candidateMovesMlp = []
		optimalOutput = 0 

		while len(setMoves) != 0:
			tmp_move = random.choice(setMoves)
			tmpBoard = deepcopy(boardToPlay)

			tmpBoard.push(tmp_move)
			shapedBoardMlp = self.shapeBoardMlp(tmpBoard)
			out = self.evaluatePositionMlp(shapedBoardMlp, MlpModel, tmp_move, tmpBoard)
			setMoves.remove(tmp_move)

			if out > optimalOutput:
				candidateMovesMlp = [] 
				optimalOutput = out
				candidateMovesMlp.append(tmp_move)
			
			elif out == optimalOutput:
				candidateMovesMlp.append(tmp_move)

		return candidateMovesMlp

	def makeCandidateMovesCnn(self, boardToPlay, setMoves, CnnModel):

		candidateMovesCnn = []
		optimalOutput = 0 

		while len(setMoves) != 0:
			tmp_move = random.choice(setMoves)

			tmpBoard = deepcopy(boardToPlay)

			tmpBoard.push(tmp_move)
			shapedBoardCnn = np.asarray(self.shapeBoardCnn(tmpBoard))

			shapedBoardCnn = np.reshape(shapedBoardCnn, (8,8,16))

			out = self.evaluatePositionCnn(shapedBoardCnn, CnnModel, tmp_move, tmpBoard)
			setMoves.remove(tmp_move)

			if out > optimalOutput:
				candidateMovesCnn = [] 
				optimalOutput = out
				candidateMovesCnn.append(tmp_move)
			
			elif out == optimalOutput:
				candidateMovesCnn.append(tmp_move)

		return candidateMovesCnn
		
	def startGame(self, boardToPlay, MlpModel, CnnModel):

		while not boardToPlay.is_game_over(claim_draw=True):

			setMoves = list(self.createSetMoves(boardToPlay))
			#MlpcandidateSetMoves = list(self.makeCandidateMovesMlp(boardToPlay, setMoves, MlpModel))
			Cnncandidatesetmoves = list(self.makeCandidateMovesCnn(boardToPlay, setMoves, CnnModel))

			move = random.choice(Cnncandidatesetmoves)	#if multiple moves have same evaluation choose a random one
	
			boardToPlay.push(move)

			print(boardToPlay)
			print "-------------------------------"
			time.sleep(0.4)

		result = boardToPlay.result()
		self.updateGameStatsWhite(result)

	def main(self):
		
		MlpModel = self.loadMlpClassificationModel()
		CnnModel = self.loadCnnClassificationModel()

		with open(self.StartingPositionsPath) as f:
			individualPositions = f.readlines()
		
		for position in individualPositions:
			boardToPlay = self.loadStartingPositions(position)
			for i in xrange(0, self.NumberSimulationGames):
				copiedBoard = deepcopy(boardToPlay)
				
				self.startGame(copiedBoard, MlpModel, CnnModel)

			print "Amount of Draws: ", Gamer.Draws
			print "Amount of MLP Wins: ", Gamer.MlpWins
			print "Amount of CNN Wins: ", Gamer.CnnWins

Gamer = GameHandler()
Gamer.main()

