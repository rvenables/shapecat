import io
import matplotlib.pyplot as plt;
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
import os

class CandidateModelAdapter():
    '''
    Converts candidate geometry into a format that can be processed by the model

    Args:
        dimension (int): The target output dimension of the converted geometry
        save_directory (str): The optional directory to save copies of saved items
    
    '''
    def __init__(self, graph, dimension=180, save_directory=False):
        
        self.graph = graph
        self.dimension = dimension
        self.save_directory = save_directory

    def build_patch(self, boundary_nodes):
        ''' Converts the target boundary node list to an new (matplotlib) path patch

            Args:
                boundary_nodes (node list): The source geometry to convert

            Returns:
                PathPatch: The converted patch from the source geometry
        '''
        xy = []

        edge_nodes = list(zip(boundary_nodes[:-1], boundary_nodes[1:]))
        
        for u, v in edge_nodes:
            data = min(self.graph.get_edge_data(u, v).values(), 
                    key=lambda x: x['length'])

            if 'geometry' in data:
                # More Complex Geometry
                xs, ys = data['geometry'].xy
                points = list(zip(xs, ys))
                xy.extend(points)

            else:
                # Straight Lines
                xy.append((self.graph.nodes[u]['x'], self.graph.nodes[u]['y']))
                xy.append((self.graph.nodes[v]['x'], self.graph.nodes[v]['y']))

        return PathPatch(Path(xy, closed=True), facecolor='black', edgecolor='black', zorder=-1)


    def convert(self, boundary_nodes, save_index=False):
        ''' Converts the target boundary node list to an numpy array (representing an image)

            Args:
                boundary_nodes (node list): The source geometry to convert
                save_index (int): A unique index corresponding to the current item, 
                             used exclusively for the (optional) output filename

            Returns:
                numpy array (dimension x dimension): The converted image from the source geometry
        '''

        patch = self.build_patch(boundary_nodes)

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.add_patch(patch)
        plt.axis('off')
        ax.axis('scaled')

        buffer = io.BytesIO()

        plt.savefig(buffer, dpi=self.dimension/2, format='raw')

        if self.save_directory and save_index is not False:
            fig.savefig(os.path.join(self.save_directory,f'{save_index}.png'))

        buffer.seek(0)

        img_arr = np.reshape(np.frombuffer(buffer.getvalue(), dtype=np.uint8),
                            newshape=(self.dimension, self.dimension, -1))

        buffer.close()
        plt.clf()
        plt.close(fig)

        return img_arr