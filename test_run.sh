#!/bin/bash

# run the commands to test the pixel writes
./scripts/compile;
echo "removing shots/*"
<<<<<<< HEAD
rm shots/*;
python3 Controller/AsteroidsController.py Asteroids/Asteroids shots/
=======
rm -r shots/*;
python3.6 Controller/AsteroidsController.py Asteroids/Asteroids shots/i
>>>>>>> origin/master

open shots/*
