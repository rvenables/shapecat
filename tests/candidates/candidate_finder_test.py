import sys, os
import osmnx as ox
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path + '/../../')

# A sample graph, matching the expectations in the test suite, is required
SAMPLE_GRAPH_PATH = 'data/test.graphml'

# Enable to create a visual trace of these unit tests
VISUAL_TRACE = False
VISUAL_TRACE_PATH = 'test-trace'

# Enable to print a trace of the graph search on failed unit tests
GRAPH_TRACE = False

from candidates.candidate_finder import CandidateFinder
from candidates.candidate_model_adapter import CandidateModelAdapter

def debug_draw_candidate(graph, candidate, unique_name):
    ''' For visual tracing, identified candidate is drawn in red '''

    if not os.path.exists(VISUAL_TRACE_PATH):
        os.makedirs(VISUAL_TRACE_PATH)

    edges = list(zip(candidate[:-1], candidate[1:]))
    ec = ['r' if (u,v) in edges or (v, u) in edges else 'grey' for u, v, k in graph.edges(keys=True)]
    ox.plot_graph(graph, show=False, save=True, filepath=os.path.join(VISUAL_TRACE_PATH,f'debug-{unique_name}.png'), edge_color=ec)

def debug_draw_candidates(graph, candidates, prefix):
    ''' For visual tracing '''

    if VISUAL_TRACE:
        i = 0
        for candidate in candidates:
            i += 1
            debug_draw_candidate(graph, candidate, f'{prefix}-{i}')

def build_sample_location():
    ''' Utility method for building a new sample location '''
    graph = ox.graph_from_point((27.7864647,-82.673489), dist=500)
    ox.plot_graph(graph, show=False, save=True, filepath='test-clean.png', close=False)
    ox.save_graphml(graph,SAMPLE_GRAPH_PATH)

def assert_correct_candidates_found(actual, expected):
    ''' Confirms all of the expected candidates were found '''

    assert(len(expected) == len(actual))

    actual_sets = [set(a[:-1]) for a in actual]

    for exp in expected:
        assert(set(exp) in actual_sets)

def get_sample_location():
    return ox.load_graphml(SAMPLE_GRAPH_PATH)

def test_finds_correct_single_rectangle_and_line_min1_max4():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=1, max_segments=4, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min1.max4')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262],
        [99823974, 99823976]])

def test_finds_correct_single_rectangle_and_line_min2_max4():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=2, max_segments=4, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min2.max4')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262],
        [99823974, 99823976]])

def test_finds_no_rectangle_because_max_length():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=3, max_segments=4, min_length_meters=10, max_length_meters=11)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-low-max')
    
    assert(len(candidates) == 0)

def test_finds_no_rectangle_because_min_length():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=3, max_segments=4, min_length_meters=999, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-high-min')
    
    assert(len(candidates) == 0)

def test_finds_correct_single_rectangle_min3_max4():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=3, max_segments=4, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min3.max4')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262]])

def test_finds_no_candidates_rectangle_min3_max3():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=3, max_segments=3, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min3.max4')
    
    assert(len(candidates) == 0)

def test_finds_correct_single_rectangle_min4_max4():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=4, max_segments=4, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min4.max4')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262]])

def test_finds_correct_two_rectangles_min4_max5():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=4, max_segments=5, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)
    debug_draw_candidates(graph,candidates,'single-rect-min4.max5')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262], 
        [99823974, 99823976, 4742319316, 4742319324, 4742319315]])


def test_finds_correct_three_rectangles_min4_max6():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=4, max_segments=6, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)

    print([i[:-1] for i in candidates])

    debug_draw_candidates(graph,candidates,'single-rect-min4.max6')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262], 
        [99823974, 99823976, 99823978, 99967264, 99948676, 99967262], 
        [99823974, 99823976, 4742319316, 4742319324, 4742319315]])

def test_finds_correct_seven_rectangles_min4_max7():

    graph = get_sample_location()
    finder = CandidateFinder(graph, min_segments=4, max_segments=7, min_length_meters=10, max_length_meters=1000)
    finder.trace_search = GRAPH_TRACE

    candidates = finder.find_candidates(40)

    print([i[:-1] for i in candidates])

    debug_draw_candidates(graph,candidates,'single-rect-min4.max7')
    
    assert_correct_candidates_found(
        candidates,
        [[99823974, 99823976, 99948676, 99967262, 99937288, 8382823734, 99823972], 
        [99823974, 99823976, 99948676, 99967262], 
        [99823974, 99823976, 99823978, 99967264, 99948676, 99967262], 
        [99823974, 99823976, 4742319316, 99901928, 4742319325, 4742319324, 4742319315], 
        [99823974, 99823976, 4742319316, 99901928, 4742319325, 99901926, 4742319315], 
        [99823974, 99823976, 4742319316, 4742319324, 4742319315], 
        [99823974, 99823976, 4742319316, 4742319324, 4742319325, 99901926, 4742319315]])