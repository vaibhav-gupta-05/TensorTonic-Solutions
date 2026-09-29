import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if max_len == None:
        max_len = max((len(i) for i in seqs) , default = 0)

    for i in seqs :
        if len(i)<max_len:
            i.extend([pad_value]*(max_len-len(i)))
        elif len(i)>max_len:
            i[:] = i[:max_len]
        else:
            continue
    arr = np.array(seqs, dtype=int)
    if arr.ndim == 1: 
        return arr.reshape(len(seqs), max_len)
        
    return arr