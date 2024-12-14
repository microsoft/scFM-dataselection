import pickle

def load_subsample_index(pickle_filename):
    """
    Loads the indices for a geometric sketch that was previously computed using
    sketch_anndata() and saved to a pickle file.
 
    Args:
        pickle_filename (str): The binary pickle file containing the sketch index.

    Returns:
        A list containing the indices of the cells found in the sketch.
    """
    with open(pickle_filename, 'rb') as pickle_file:
        sketch_index = pickle.load(pickle_file)
    return sketch_index
