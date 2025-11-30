#!/bin/bash

args="$@"
echo "Running train.py with arguments: $args"
python train.py $args && python test.py $args