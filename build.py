'''
                                                                                                                                            
 .@@@@%                                                     
      ,@@@                                                  
        (@@                                                 
          @@@                                      @#   @   
            @@@@@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
               @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
                &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&..
                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@             
                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@              
              @@@@@@@@   @@@@      @@@@@@@@&                
              @@@         @@@        @@@@@@@@               
              @@.          @@        @@@ #@@@@@             
              @@@          @@@.       @@%   .@@@@           
                                       @@@.    @@@@         
                                                 (@        
    [shapecat] Model Build Tool
    Run Before Using Search
    https://github.com/rvenables/heart-search
    Copyright 2021 Rob Venables
'''

import argparse
import matplotlib
import numpy as np
import os
import osmnx as ox
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from candidates.candidate_finder import CandidateFinder
from candidates.candidate_model_adapter import CandidateModelAdapter
from model.character_example_builder import CharacterExampleBuilder
from model.model_builder import ModelBuilder

import tensorflow as tf

# Attempted correction for intermittent issue with CUDNN_STATUS_ALLOC_FAILED (on Windows)
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu and len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

parser = argparse.ArgumentParser(description='Heart Search Model Build Tool')
parser.add_argument("--character", default="♥", required=False, help="character/emoji for model construction")
parser.add_argument("--save", default=False, required=False, help="save copies of training images")
parser.add_argument("--savedir", default='./train', required=False, help="training image storage location (only relevant if save=true)")
parser.add_argument("--size", default=100, type=int, required=False, help="the number of positive and negative examples (each)")
parser.add_argument("--model", default="model.h5", required=False, help="the name of the output model")
parser.add_argument("--city", default="Sarasota, FL, USA", required=False, help="the name of the city for negative training data")
parser.add_argument("--dim", default=180, type=int, required=False, help="the width/height of the training images (larger=slower)")
args = parser.parse_args()

matplotlib.use('Agg')
ox.config(use_cache=True)

est_memory_demand_mb = int(((args.size ** 2) * 8 * 2) / 1024 ** 2)
print(f'{args.size:,} Examples Requested, At Least {est_memory_demand_mb*2:,}MB System Memory Needed ({est_memory_demand_mb*2:,}MB GPU).')

label_dir_pos = os.path.join(args.savedir,'1') if args.save else False
label_dir_neg = os.path.join(args.savedir,'0') if args.save else False

if args.save:
    os.makedirs(label_dir_pos, exist_ok=True)
    os.makedirs(label_dir_neg, exist_ok=True)

print(f'Loading City ({args.city}) for Negative Training Data...')

graph = ox.graph_from_place(args.city)

positives = np.zeros((args.size,args.dim,args.dim, 1))
negatives = np.zeros((args.size,args.dim,args.dim, 1))

builder = CharacterExampleBuilder(args.character, args.dim, label_dir_pos)

# Generate Positive Examples
for i in tqdm(range(args.size), desc="Building Positive Examples"):
    img_arr = builder.generate()
    positives[i] = img_arr[:,:,0:1]

# Generate Negative Examples
# Based on City (args.city)
# Key Assumption: No Geometry is Exact Match

finder = CandidateFinder(graph)
adapter = CandidateModelAdapter(graph,args.dim, label_dir_neg)
total_candidates = 0

with tqdm(total=args.size,desc="Building Negative Examples") as pbar:
    while (total_candidates < args.size):
        edge_index = finder.get_random_index()
        new_candidates = finder.find_candidates(edge_index)
        
        for candidate in new_candidates:
            img_arr = adapter.convert(candidate, total_candidates)
            negatives[total_candidates] = img_arr[:,:,0:1]
            total_candidates += 1
            pbar.update(1)

            if total_candidates == args.size:
                break

model_builder = ModelBuilder(args.dim, args.model)
model_builder.report_accuracy = True
model_builder.build(positives, negatives)