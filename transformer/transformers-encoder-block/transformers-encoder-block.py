import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q,K,V,W_q,W_k,W_v,W_o,num_heads):
    """
    Returns multi-head attention output.

    Expected shapes:
        Q, K, V : (batch_size, seq_length, d_model)
        W_q     : (d_model, d_model)
        W_k     : (d_model, d_model)
        W_v     : (d_model, d_model)
        W_o     : (d_model, d_model)

    Returns:
        Output : (batch_size, seq_length, d_model)
    """

    batch_size,seq_len,d_model=Q.shape

    assert d_model % num_heads == 0, \
        "d_model must be divisible by num_heads"

    d_head = d_model // num_heads

    #Linear Projection
    
    Q_new = Q @ W_q
    K_new = K @ W_k
    V_new = V @ W_v

    # split into multiple head
    # for eq x in shape of (1,3,4) --> (1,3,2,2)
    Q_heads=Q_new.reshape(batch_size,seq_len,num_heads,d_head)
    K_heads=K_new.reshape(batch_size,seq_len,num_heads,d_head)
    V_heads=V_new.reshape(batch_size,seq_len,num_heads,d_head)

    # Move heads before sequence dimension
    # Before:(batch, seq_length, num_heads, d_head)
    # After:(batch, num_heads, seq_length, d_head)
    Q_heads=Q_heads.transpose(0,2,1,3)
    K_heads=K_heads.transpose(0,2,1,3)
    V_heads=V_heads.transpose(0,2,1,3)


    # Scaled dot-product attention Q.K.T
    # K_heads.transpose(-2, -1):
    # (batch, heads, seq_length, d_head)
    #                    ↓
    # (batch, heads, d_head, seq_length)
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    # Scale by sqrt(d_head)
    scores = scores / np.sqrt(d_head)

    # Softmax over keys
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of V

    head_outputs = attention_weights @ V_heads

    # Shape:
    # (batch, heads, seq_length, d_head)

    # Move dimensions back

    head_outputs = head_outputs.transpose(0, 2, 1, 3)

    # Shape:
    # (batch, seq_length, heads, d_head)

    # Concatenate all heads

    head_outputs = head_outputs.reshape(
        batch_size, seq_len, d_model
    )
    # Output projection
    output = head_outputs @ W_o

    return output

def residual(x,att_x):
    return x+att_x

def layer_norm(x, gamma, beta, eps=1e-6):
    mean=np.mean(x,axis=-1,keepdims=True)
    var=np.var(x,axis=-1,keepdims=True)
    x_norm=(x-mean)/np.sqrt(var+eps)
    output=(gamma*x_norm)+beta
    return output

def relu(x):
    return np.maximum(0, x)

def feed_forward(x,W1,b1,W2,b2):
    h1 = x @ W1
    h1 = h1 + b1
    h1 = relu(h1)
    out = h1 @ W2
    out = out + b2
    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    # calculate dim_model.
    
    multi=multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    first_resi=residual(x,multi)
    fisrt_layer=layer_norm(first_resi,gamma1,beta1)
    # layer_norm(x, gamma, beta, eps=1e-6)
    feed_fn=feed_forward(fisrt_layer,W1,b1,W2,b2)
    second_resi=residual(fisrt_layer,feed_fn)
    second_layer=layer_norm(second_resi,gamma2,beta2)

    return second_layer
    
    