def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """

    hit=0
    for i in range(k):
        if recommended[i] in relevant:
            hit+=1
            
    p=hit/k
    r=hit/len(relevant)
    
    return [p,r]
    # Write code here