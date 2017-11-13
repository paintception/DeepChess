import numpy as np

def read_data():

	X = []
	y = []

        with open(".txt", 'r') as f:	# Location of txt file

		print "Reading the Data"
		try:
			for line in f:
				record = line.split(";")
				pieces = [eval(x) for x in record[0:12]]
				piece = [item for sublist in pieces for item in sublist]
				piece = [item for sublist in piece for item in sublist]	

				features = [eval(x) for x in record[12:16]]
				feature = [item for sublist in features for item in sublist]

				Information = np.asarray(piece + feature)
				Evaluation = float(record[16][:-2])

				X.append(Information)
				y.append(Evaluation)

		except:
			pass

	X = np.asarray(X)
	NewX = X.reshape((len(X), 8, 8, 16))
	
	new_y = []

	for evaluation in y:
		if evaluation >= - 1.5 and evaluation <= 1.5:
			new_y.append("Equal")
		elif evaluation > 1.5:
			new_y.append("WhiteWinning")
		elif evaluation < -1.5:
			new_y.append("BlackWinning")
		
	new_y = np.asarray(new_y)

	print len(NewX)
	print len(new_y)

	np.save(STORING_PATH+'CnnFeatureInput.npy', NewX)
	np.save(STORING_PATH+'CnnFeatureInput.npy', new_y)

def main():
        read_data()
       
if __name__ == '__main__':
	main()
