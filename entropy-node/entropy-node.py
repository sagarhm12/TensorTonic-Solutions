import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Get unique class frequencies
    _, counts = np.unique(y, return_counts=True)
    
    # Perfect purity (0 or 1 class) means 0 entropy
    if len(counts) <= 1:
        return 0.0
    
    # Convert absolute counts to class probabilities
    probabilities = counts / np.sum(counts)
    print(probabilities)
    # Compute Shannon Entropy with a tiny epsilon value for numerical stability
    epsilon = 1e-15
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy
