import networkx as nx
import random
import hashlib

class CandidateFinder():
    '''
    Identifies candidate geometry from a graph, given a set of constraints.

    Candidate geometry is constructed as a list of nodes in the source graph.

    The most important constructor parameter is the maximum segment count. Higher values for this
    parameter will yield more complex geometry and more candidates, at the cost of additional
    search time.

    Args:
        graph (osmnx graph): The source graph to analyze
        max_segments (int): The maximum amount of segments to consider.
        min_segments (int): The minimum amount of segments to consider.
        max_length_meters (int): The maximum length of candidate routes, in meters.
        min_length_meters (int): The minimum length of candidate routes, in meters.
    
    '''
    def __init__(self, graph, max_segments=10, min_segments=4, max_length_meters=10000, min_length_meters=750):
        
        self.graph = self.preprocess_graph(graph)
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.min_length_meters = min_length_meters
        self.max_length_meters = max_length_meters

        self.total_edges = len(self.graph.edges)
        self.edge_options = list(self.graph.edges)

        # Enable to allow for tracing of the search feature to the console
        self.trace_search = False

    def preprocess_graph(self, graph):
        ''' Cleans up the graph by removing self-loops. Faciliated by k_core
            https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.k_core.html

            Args:
                graph (osmnx graph): The source graph to clean up

            Returns:
                osmnx graph: The processed graph
        '''
        nx_graph = nx.Graph(graph, as_view=True)
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
        core_nodes = nx.k_core(nx_graph, 2)
        return graph.subgraph(core_nodes)

    def get_random_index(self):
        ''' Returns a valid random edge index

            Returns:
                int: A random edge index in the range of zero to total_edges-1
        '''

        return random.randint(0,self.total_edges-1)

    def get_points(self, candidate):
        ''' Converts a candidate into a list of latitude and longitudes
        
            Args:
                candidate (list): A list of intersections representing a candidate.

            Returns:
                list: A list of latitude and longitudes representing the candidate
        '''

        points = []

        for ist in candidate:
            points.append((self.graph.nodes[ist]["y"], self.graph.nodes[ist]["x"]))

        return points

    def build_map_link(self, candidate):
        ''' Converts a candidate into a new Google Maps link
        
            Args:
                candidate (list): A list of intersections representing a candidate.

            Returns:
                list: A list of latitude and longitudes representing the candidate
        '''
        points = self.get_points(candidate)
        base_url = 'https://www.google.com/maps/dir/'
        return base_url + '/'.join([f'{i[0]},{i[1]}' for i in points])

    def get_id(self, candidate):
        ''' Converts a candidate into a semi-unique short id. 
            Collision risk is present but utility appears higher.
        
            Args:
                candidate (list): A list of intersections representing a candidate.

            Returns:
                list: A semi-unique id representing the candidate
        '''

        points = self.get_points(candidate)
        shape = '/'.join([f'{i[0]},{i[1]}' for i in sorted(points)])
        hash = hashlib.sha1(shape.encode("UTF-8")).hexdigest()
        return hash[:16]

    def find_loops(self, path, path_distance):
        ''' Identifies closed loops, given a starting path. Operates recursively.
        
            Args:
                path (list): A list of intersections representing a candidate.
                path_distance (int): A precalculated length of the existing path, minus the last node

            Returns:
                list: A list of valid candidate loops
        '''

        start = path[-2]
        stop = path[-1]

        num_segments = len(path) - 1

        arrow = f'{"".join(["---" for i in path])}>'

        self.trace(f'{arrow} Evaluating {stop} (At Length {num_segments})')

        data = min(self.graph.get_edge_data(start, stop).values(), 
                    key=lambda x: x['length'])
        length = data["length"]

        self.trace(f'{arrow} {self.graph.nodes[stop]["y"]}, {self.graph.nodes[stop]["x"]}')

        new_distance = path_distance + length

        if num_segments > self.max_segments:
            self.trace(f'{arrow} Beyond Segment Length ({num_segments} > {self.max_segments})')
            return []

        if new_distance > self.max_length_meters:
            self.trace(f'{arrow} Beyond Max Length Meters')
            return []

        if num_segments > 1:
            if stop in path[:-1]:

                if stop == path[0]:
                    # (Arrived back at start)
                    self.trace(f'{arrow} Found Start')

                    if new_distance < self.min_length_meters:
                        self.trace(f'{arrow} new distance too small {new_distance}m...')
                        return []

                    # Enforce minimum segment requirement
                    if num_segments < self.min_segments:
                        self.trace(f'{arrow} New distance too few segments {num_segments} segments in {new_distance}m...')
                        return []

                    # Appears to be a valid loop
                    self.trace(f'{arrow} Valid Solution Identified.')
                    return [path]

                else:
                    # Candidates should not intersect existing path
                    self.trace(f'{arrow} Hits Existing Path')
                    return []

        candidates = set(self.graph.neighbors(stop))

        if not candidates:
            self.trace(f'{arrow} No Candidates')
            return []
        
        paths = []

        for candidate in candidates:

            self.trace(f'{arrow} Evaluating Candidate {candidate}...')

            new_path = path.copy()
            new_path.append(candidate)
            
            found_paths = self.find_loops(new_path, new_distance)
            paths.extend(found_paths)

        return paths


    def find_candidates(self, starting_edge_index):
        '''Returns all candidates from starting index, subject to configured constraints.

        Args:
            starting_edge_index (int): The index of the intersection to start analyzing

        Returns:
            list: A list of lists representing candidate geometry intersecting the starting node 
        '''
        start, stop, _ = self.edge_options[starting_edge_index]
        return self.find_loops([start, stop], 0)

    def trace(self, message):
        if self.trace_search:
            print(message)