# Script that labels different board positions according to Stockfish's evaluation
# function and creates suitable chess board representations for the ANNs

from __future__ import division

import time
import chess
import chess.pgn
import chess.uci
import numpy as np
import os

GAMES_DIRECTORY = ''	# Directory where the pgn games are saved
STORING_PATH = ''		# Where to store the games evaluated by Stockfish

engine = chess.uci.popen_engine('/usr/games/stockfish')	# Be sure to have Stockfish installed
engine.uci()

ImportantSquareSet = chess.SquareSet(

	chess.BB_D4 | chess.BB_D5 |
	chess.BB_C4 | chess.BB_C5 |
	chess.BB_E4 | chess.BB_E5 |
	chess.BB_F2 | chess.BB_F7 | 
	chess.BB_H2 | chess.BB_H7

	)

SquareSet = chess.SquareSet(

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

def load_game():

	for root, dirs, filenames in os.walk(GAMES_DIRECTORY):
		for f in filenames:
			if f.endswith('.pgn'):
				try:
					pgn = open(os.path.join(root, f), 'r')
					game = chess.pgn.read_game(pgn)
					process_game(game)
					os.remove(GAMES_DIRECTORY+str(f))
				except:
					pass

def splitter(inputStr, black):
	
	inputStr = format(inputStr, "064b")
	tmp = [inputStr[i:i+8] for i in range(0, len(inputStr), 8)]
	
	for i in xrange(0, len(tmp)):
		tmp2 = list(tmp[i])
		tmp2 = [int(x) * black for x in tmp2]
		tmp[i] = tmp2

	return tmp

def is_checked(board):

	if board.is_check() and board.turn is True:
		CheckedInfo = [-1] * 64
	
	elif board.is_check() and board.turn is False:
		CheckedInfo = [1] * 64
	
	elif not board.is_check():
		CheckedInfo = [0] * 64
	
	return CheckedInfo

def MlpBitmaps(board, e, filename):
	
	P_input = splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
	R_input = splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)
	N_input = splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
	B_input = splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
	Q_input = splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)
	K_input = splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

	p_input = splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
	r_input = splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
	n_input = splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
	b_input = splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
	q_input = splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
	k_input = splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

	with open(filename, 'a') as thefile:

		thefile.write("%s;" % P_input)
		thefile.write("%s;" % R_input)
		thefile.write("%s;" % N_input)
		thefile.write("%s;" % B_input)
		thefile.write("%s;" % Q_input)
		thefile.write("%s;" % K_input)
		thefile.write("%s;" % p_input)
		thefile.write("%s;" % r_input)
		thefile.write("%s;" % n_input)
		thefile.write("%s;" % b_input)
		thefile.write("%s;" % q_input)
		thefile.write("%s;" % k_input)
		thefile.write("%s\n" % e)

def CnnBitmaps(board, e, filename):
	
	P_input = splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
	R_input = splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)
	N_input = splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
	B_input = splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
	Q_input = splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)
	K_input = splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

	p_input = splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
	r_input = splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
	n_input = splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
	b_input = splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
	q_input = splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
	k_input = splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

	CheckedInfo = is_checked(board)

	SquareAttackers = []
	PinnedSquares = []

	ImportantAttackers = []

 	for square in SquareSet:
		if board.is_attacked_by(chess.WHITE, square):
			SquareAttackers.append(1)
		elif board.is_attacked_by(chess.BLACK, square):
			SquareAttackers.append(-1)
		else:
			SquareAttackers.append(0) 
	
		if board.is_pinned(chess.WHITE, square):
			PinnedSquares.append(1)
		elif board.is_attacked_by(chess.BLACK, square):
			PinnedSquares.append(-1)
		else:
			PinnedSquares.append(-1)

	attackers_tracker = []

	for ImportantSquare in ImportantSquareSet:
		WhiteAttackers = board.attackers(chess.WHITE, ImportantSquare)
		BlackAttackers = board.attackers(chess.BLACK, ImportantSquare)
		
		if len(WhiteAttackers) > len(BlackAttackers):
			ImportantAttackersFeatures = [1] * 64
		elif len(WhiteAttackers) < len(BlackAttackers):
			ImportantAttackersFeatures = [-1] * 64
		else:
			ImportantAttackersFeatures = [0] * 64

	with open(filename, 'a') as thefile:

		thefile.write("%s;" % P_input)
		thefile.write("%s;" % R_input)
		thefile.write("%s;" % N_input)
		thefile.write("%s;" % B_input)
		thefile.write("%s;" % Q_input)
		thefile.write("%s;" % K_input)
		thefile.write("%s;" % p_input)
		thefile.write("%s;" % r_input)
		thefile.write("%s;" % n_input)
		thefile.write("%s;" % b_input)
		thefile.write("%s;" % q_input)
		thefile.write("%s;" % k_input)
		thefile.write("%s;" % CheckedInfo)
		thefile.write("%s;" % SquareAttackers)
		thefile.write("%s;" % PinnedSquares)
		thefile.write("%s;" % ImportantAttackersFeatures)
		thefile.write("%s\n" % e)

def GameChecker(board):
	P_input = splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
	R_input = splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 5)
	N_input = splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 3)
	B_input = splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 3)
	Q_input = splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 9)

	p_input = splitter(int(board.pieces(chess.PAWN, chess.BLACK)), 1)
	r_input = splitter(int(board.pieces(chess.ROOK, chess.BLACK)), 5)
	n_input = splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), 3)
	b_input = splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), 3)
	q_input = splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), 9)

	Status = P_input+R_input+N_input+B_input+Q_input+p_input+r_input+n_input+b_input+q_input
	TmpStatus = [item for sublist in Status for item in sublist]

	return sum(TmpStatus)

def makeDatasets(board, evaluation, moveCnt):

	if moveCnt < 40:	# Before the 40 plys we consider the moves that have been
						# played part of the opening

		MlpBitmaps(board,evaluation,STORING_PATH+'MlpFileOpening.txt')
		CnnBitmaps(board,evaluation,STORING_PATH+'CnnFileOpening.txt')
	else:
		cp = GameChecker(board)
		if cp <= 12:	# If the value of the pieces on the Board is smaller
						# than 12 we assume we are in the endgame stage

			MlpBitmaps(board,evaluation,STORING_PATH+'MlpFileEnd.txt')
			CnnBitmaps(board,evaluation,STORING_PATH+'CnnFileEnd.txt')

		elif cp > 12:	# Otherwise we are still in the MiddleGame
			
			MlpBitmaps(board,evaluation,STORING_PATH+'MlpFileMiddle.txt')
			CnnBitmaps(board,evaluation,STORING_PATH+'CnnFileMiddle.txt')
		else:
			pass

def process_game(game):

	positions = []
	evaluations = []

	GM_board = chess.Board()
	node = game
	movetime = 100	#Milliseconds, the lower the more approximate Stockfish's evaluation is

	info_handler = chess.uci.InfoHandler()
	engine.info_handlers.append(info_handler)

	tmp = 0

	while not node.is_end():

		try:
			engine.position(GM_board)
			b_m = engine.go(movetime=movetime)

			info = info_handler.info["score"][1]
		
			next_node = node.variation(0)
		
			if info[0] is not None and GM_board.turn is True:				
				stock_evaluation = info[0]/100 
				new_stock_evaluation = stock_evaluation 
						
				GM_move = str(node.board().san(next_node.move))
				GM_board.push_san(GM_move)
				makeDatasets(GM_board, new_stock_evaluation, tmp)
			
			elif info[0] is not None and GM_board.turn is False:
				
				stock_evaluation = info[0]/100
				stock_evaluation = - stock_evaluation # Flip evaluations for Black
						
				GM_move = str(node.board().san(next_node.move))
				GM_board.push_san(GM_move)
				makeDatasets(GM_board, stock_evaluation, tmp)
					
			node = next_node
			tmp = tmp + 1

		except:
			print('Unknown Position')
			pass

if __name__ == '__main__':
	print('Job has started')	
	load_game()
