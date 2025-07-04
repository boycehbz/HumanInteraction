import numpy as np

COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0]]
CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]]] 

COLORS_NON_CONTACT = np.zeros((2, 6890, 4))
COLORS_NON_CONTACT[0,:] = np.array([0.412,0.663,1.0,1.0])
COLORS_NON_CONTACT[1,:] = np.array([1.0,0.749,0.412,1.0])

