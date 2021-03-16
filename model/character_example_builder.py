import io
import math
import matplotlib.pyplot as plt;
import numpy as np
import os
import random

class CharacterExampleBuilder():
    '''
    Constructs positive examples to create training data, based on single characters or emoji

    Args:
        character (str): The source character to use to build positive examples
        dimension (int): The maximum amount of segments to consider.
        save_directory (str): The optional directory to save copies of examples
    
    '''
    def __init__(self, character='♥', dimension=180, save_directory=False):
        
        self.character = character
        self.dimension = dimension
        self.save_directory = save_directory

        np.random.seed(42)
    
        self.rotations_10x = range(0,3600)

        self.loc_xs = np.random.normal(dimension/330, 0.01, 100)
        self.loc_ys = np.random.normal(dimension/515, 0.01, 100)
        self.sizes = np.random.normal(dimension/1.64, dimension/18, 1000)

    def generate(self):
        ''' Builds a single positive training example. Call this in a loop to generate many.

            Returns:
                numpy array (dimension x dimension): The new training example
        '''

        fig, _ = plt.subplots(figsize=(2, 2))

        x = random.choice(self.loc_xs)
        y = random.choice(self.loc_ys)
        size = random.choice(self.sizes)
        rotation = random.choice(self.rotations_10x) / 10

        # Basic compensation for off-center rotational axis
        y_off = math.sin(rotation/110)/4
        x_off = math.sin(rotation/180)/8

        plt.text(x-x_off, y+y_off, self.character, fontsize=size,
                rotation=rotation, rotation_mode='anchor', ha='center', va='center', linespacing=1)

        plt.axis('off')
        buffer = io.BytesIO()

        plt.savefig(buffer, dpi=self.dimension/2, format='raw')

        if self.save_directory:
            fig.savefig(os.path.join(self.save_directory,f'{rotation}_{x}_{y}_{size}.png'))

        buffer.seek(0)

        img_arr = np.reshape(np.frombuffer(buffer.getvalue(), dtype=np.uint8),
                            newshape=(self.dimension, self.dimension, -1))

        buffer.close()
        plt.clf()
        plt.close(fig)

        return img_arr
