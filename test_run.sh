#!/bin/bash

# run the commands to test the pixel writes
./compile;
echo "removing shots/*"
rm shots/*;
python3.6 Controller/AsteroidsController.py Asteroids/Asteroids shots/i

open shots/*
