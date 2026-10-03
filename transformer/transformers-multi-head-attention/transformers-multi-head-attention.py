import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
    W_o: np.ndarray, num_heads: int
) -> np.ndarray:

    # 1. Linear projections
    q_new = Q @ W_q
    k_new = K @ W_k
    v_new = V @ W_v

    # 2. Calculate dimension of each head
    d_model = q_new.shape[-1]
    d_head = d_model // num_heads

    # 3. Store outputs of each head
    head_outputs = []

    # 4. Process each head
    for h in range(num_heads):

        start = h * d_head
        end = start + d_head

        # Split Q, K, V for this head
        q_head = q_new[:, :, start:end]
        k_head = k_new[:, :, start:end]
        v_head = v_new[:, :, start:end]

        # 5. Scaled dot-product attention
        scores = q_head @ np.swapaxes(k_head, -1, -2)

        scores = scores / np.sqrt(d_head)

        # 6. Softmax
        attention = softmax(scores, axis=-1)

        # 7. Weighted sum of V
        head_output = attention @ v_head

        head_outputs.append(head_output)

    # 8. Concatenate all heads
    result = np.concatenate(head_outputs, axis=-1)

    # 9. Output projection
    output = result @ W_o

    return output