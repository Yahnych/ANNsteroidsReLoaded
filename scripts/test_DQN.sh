#!/bin/bash

# run the commands to test the pixel writes
./scripts/compile;
echo "removing shots/*"

rm -r shots/*;
/anaconda/bin/python Controller/DQNcontroller.py Asteroids/Asteroids shots/;
