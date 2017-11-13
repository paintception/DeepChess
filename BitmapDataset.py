import numpy as np
import time

STORING_PATH = " "	# Where to store the Datasets for the ANNs 

def MakeBitmapDataset():

	X = []
	y = []

	with open(".txt", 'r') as f:	# TXT file with the parsed games
		for line in f:
			try:
				record = line.split(";")
				pieces = [eval(x) for x in record[0:12]]
				piece = [item for sublist in pieces for item in sublist]
				piece = [item for sublist in piece for item in sublist]

				X.append(piece)
				y.append(float(record[12][:-2]))
			except:
				pass

	print(X)
	print(y)

	new_y = []
	Pos_X = []
	
	for pos, evaluation in zip(X,y):

		# Example that shows how to create Dataset 1
		# Add more statements for reproducing the other Datasets 

		if evaluation > -1.5 and evaluation <= 1.5:
			Pos_X.append(pos)
			new_y.append("Draw")
		
		elif evaluation < -1.5:
			Pos_X.append(pos)
			new_y.append("WinningForBlack")
		
		elif evaluation > 1.5:
			Pos_X.append(pos)
			new_y.append("WinningForWhite")
			
	print len(Pos_X)
	print len(new_y)

	np.save(STORING_PATH+'Positions.npy', X)
	np.save(STORING_PATH+'Labels.npy', y)
 
def main():
	MakeBitmapDataset()

if __name__ == '__main__':
	main()
