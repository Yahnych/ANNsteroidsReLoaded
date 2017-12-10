#!/bin/bash

# run the commands to test the pixel writes
./scripts/compile;
echo "removing shots/*"

rm shots/*;
python3 Controller/AsteroidsController.py Asteroids/Asteroids shots/


open shots/*
