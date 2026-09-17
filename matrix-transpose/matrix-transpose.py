import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    row = len(A)
    col = len(A[0])
    ans = np.zeros((col , row) , dtype = int)
    for i in range(row) :
        for j in range(col) :
            ans[j][i]  = A[i][j] 
    return ans
        
    # Write code here
    pass
