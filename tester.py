import numpy as np

las = [[1,2,3,4],
       [5,6,7,8],
       [9,1,5,2],
       [6,2,1,8]
]

las = np.array(las)
# np.put(las, [0, 3, 7], 40)
np.put(las, np.where(las == 2), 40)
print(np.where(las == 2))

# np.random.shuffle(las)
print(las)