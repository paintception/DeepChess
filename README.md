# DeepChess: **Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks**

This directory contains the basic scripts that are necessary to reproduce the results of my MSc. Artificial Intelligence Thesis entitled: **Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks**

A scientific article about my work has been presented in January 2018 at the International Conference on Pattern Recognition Applications and Methods in Madeira, Portugal. The paper can be found here http://www.scitepress.org/PublicationsDetail.aspx?ID=xWk5QRREnQk=&t=1

while it is possible to download my entire thesis from here: https://www.researchgate.net/publication/321028267_Learning_to_Play_Chess_with_Minimal_Lookahead_and_Deep_Value_Neural_Networks

or here:

http://www.ai.rug.nl/~mwiering/Thesis_Matthia_Sabatelli.pdf

The basic steps that are required to be able to reproduce my results are the following:

1. Parse a large set of chess games played by highly ranked players that you can download from here http://www.ficsgames.org/download.html with the pgnsplitter.py script  
2. Label the positions present in the individual games with the evaluation function of Stockfish and create appropriate board representations for the ANNs with the DatasetCreator.py script
3. Create the different Datasets that are reported in my MSc Thesis with either the BitmapDataset.py script if you want to train a MLP or with the FeatureDataset.py script if you want to train a CNN
4. Train the ANNs as shown in the ExampleCNN.py class. Check my MSc Thesis for the best set of hyperparameters
5. Once you have trained the ANNs let them compete agains eachother as shown in ANNsCompetition.py

**Feel free to contact me for any questions or for the weights of the ANNs :)**
