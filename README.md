# DeepChess: **Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks**

This directory contains the basic scripts that are necessary to reproduce the results of my MSc. Artificial Intelligence Thesis
entitled: **Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks**

The game of chess has always been a very important testbed for the Artificial Intelligence
community. Even though the goal of training a program to play as good as the strongest
human players is not considered as a hard challenge anymore, so far no work has been done
in creating a system that does not have to rely on expensive lookahead algorithms to play the
game at a high level. In this work we show how carefully trained Value Neural Networks are
able to play high level chess without looking ahead more than one move.
To achieve this, we have investigated the capabilities that Artificial Neural Networks (ANNs)
have when it comes to pattern recognition, an ability that distinguishes chess Grandmasters
from the more amateur players.   We firstly propose a novel training approach specifically
designed for pursuing the previously mentioned goal.  Secondly, we investigate the perfor-
mances of both Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs)
as optimal neural architecture in chess.  After having assessed the superiority of the first ar-
chitecture, we propose a novel input representation of the chess board that allows CNNs to
outperform MLPs for the first time as chess evaluation functions.  We finally investigate the
performances of our best ANNs on a state of the art test, specifically designed to evaluate the
strength of chess playing programs.  Our results show how it is possible to play high qual-
ity chess only with Value Neural Networks, without having to rely on techniques involving
lookahead.

It is possible to download my entire thesis from here: http://www.ai.rug.nl/~mwiering/Thesis_Matthia_Sabatelli.pdf

The basic steps that are required to be able to reproduce my results are the following:

1. Parse a large set of chess games played by highly ranked players with the pgnsplitter.py script  
2. Label the positions present in the individual games with the evaluation function of Stockfish and create appropriate board representations for the ANNs with the DatasetCreator.py script
3. Create the different Datasets that are reported in my MSc Thesis with either the BitmapDataset.py script if you want to train a MLP or with the FeatureDataset.py script if you want to train a CNN
4. Train the ANNs as shown in the ExampleCNN.py class. Check my MSc Thesis for the best set of hyperparameters
5. Once you have trained the ANNs let them compete agains eachother as shown in ANNsCompetition.py

**Feel free to contact me for any questions or for the weights of the ANNs :)**
**A paper about my work will be published soon, hence if you want to use my work feel free to do so but please either cite my paper as it will be published (January) or cite directly my Thesis **
