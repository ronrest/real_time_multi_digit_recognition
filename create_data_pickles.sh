#!/usr/bin/env bash

stdbuf -oL python create_data_pickles.py -i "data" | tee -a "logs/prepare_data.log"
