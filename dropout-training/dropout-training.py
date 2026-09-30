import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.array(x)

    if rng is None:
        rng = np.random.default_rng(seed = 123)

    if p==0.0:
        pattern = np.ones(x.shape ,dtype = np.int32)
        return x , pattern

    else:
        pattern = (rng.random(x.shape)>=p).astype(np.int32)
        output = (x*pattern) / (1.0-p)
        pattern = (pattern) / (1.0-p)
        return output , pattern
        
    
    
    # Write code here
    pass