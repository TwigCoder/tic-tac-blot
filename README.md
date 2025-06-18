# Tic-Tac-Blot

This project will utilize a Raspberry Pi 5 with the Camera Module 3 and a Blot from Hack Club.

The goal is to have the Raspberry Pi camera take a snapshot of the game board, pass it through an AI (well, two) that I trained to convert it into a nested list the program can understand and determine the move it should make. Then, it converts the move into commands that will be sent to the blot for it perform, ultimately drawing the move on the piece of paper. 

# Training Data

I drew a bunch of tic-tac-toe boards on blank sheets of paper, randomly drew X's and O's, took their pictures, converted them into JPEG (the default of libcamera), and threw it into the images folder. The expected output is a 3 by 3 array where 0 represents an empty spot, 1 represents an X, and 2 represents an O.