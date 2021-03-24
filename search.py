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
    [shapecat] Model Search
    Run After Using Build Tool
    https://github.com/rvenables/heart-search
    Copyright 2021 Rob Venables
'''

import argparse
import numpy as np
import os
import matplotlib
import osmnx as ox
from tqdm import tqdm
from PIL import Image
import csv

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

parser = argparse.ArgumentParser(description='[shapecat] Search Tool')
parser.add_argument("city", nargs=1, help="the name of the location to search in")
parser.add_argument("--threshold", default=0.5, type=float, required=False, help="threshold for matching images (0-1)")
parser.add_argument("--model", default="model.h5", required=False, help="the name of the input model")
parser.add_argument("--result_csv", default="matches.csv", required=False, help="the name of the results file to store matches")
parser.add_argument("--result_dir", default="matches", required=False, help="the name of the results directory to store matches")
parser.add_argument("--smin", default=4, type=int, required=False, help="the minimum number of segments to match")
parser.add_argument("--smax", default=10, type=int, required=False, help="the maximum number of segments to match")
parser.add_argument("--dmin", default=500, type=int, required=False, help="the minimum distance to match, in meters")
parser.add_argument("--dmax", default=10000, type=int, required=False, help="the maximum distance to match, in meters")
parser.add_argument("--visualize", default=True, type=bool, required=False, help="controls image saving to [result_dir] directory")
args = parser.parse_args()

if not os.path.isfile(args.model):
    print(f'Unable to find model file ({args.model}). Try running build.py first.')
    exit()

model = tf.keras.models.load_model(args.model)

# Automatically work out the model input file size (assumes square images)
dim = model.layers[0].get_output_at(0).get_shape().as_list()[1]+2

matplotlib.use('Agg')
ox.config(use_cache=True)

print(f'Loading City ({args.city})...')
graph = ox.graph_from_place(args.city)

print(f'Starting Search in {args.city} with {args.model} with a match threshold of {args.threshold}...')

finder = CandidateFinder(graph,max_segments=args.smax,
     min_segments=args.smin, max_length_meters=args.dmax, min_length_meters=args.dmin)

adapter = CandidateModelAdapter(graph,dim)

found_ids = []

if args.visualize and not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)

if not os.path.isfile(args.result_csv):
    fields=['id','probability','city','coordinates','url']
    with open(args.result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
else:
    with open(args.result_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip headers

        for row in reader:
            if len(row) > 0:
                found_ids.append(row[0])

if len(found_ids) > 0:
    print(f'Loaded {len(found_ids)} Previously Found ID(s) to Ignore...')

with open(args.result_csv, 'a', newline='') as f:
    writer = csv.writer(f)

    for i in tqdm(range(finder.total_edges),desc="Searching"):

        new_candidates = finder.find_candidates(i)

        if new_candidates:
            
            input = np.array([adapter.convert(candidate)[:,:,0:1] 
                for candidate in new_candidates])

            output = [i[0] for i in model.predict(input, batch_size=32)]

            matching_indexes = [i for i,x in enumerate(output) if x > args.threshold]

            for matching_index in matching_indexes:
                if id not in found_ids:
                    item = new_candidates[matching_index]
                    point_raw = finder.get_points(item)[0]
                    point_formatted = f'{point_raw[0]}, {point_raw[1]}'
                    link = finder.build_map_link(item)
                    id = finder.get_id(item)
                    p = format(output[matching_index], '.7f')
                    writer.writerow([id,p,args.city,point_formatted,link])
                    f.flush()
                    found_ids.append(id)

                    if args.visualize:
                        v_path = os.path.join(args.result_dir,f'{id}.png')
                        im = Image.fromarray(input[matching_index].reshape(dim,dim))
                        im.save(v_path)


        