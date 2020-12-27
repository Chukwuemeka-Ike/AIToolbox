import numpy as np

las = [[-1,2,-3,-4],
       [5,6,7,8],
       [9,1,-5,2],
       [6,2,1,8]
]

las = np.array(las)
# np.put(las, [0, 3, 7], 40)


# np.random.shuffle(las)
print(np.maximum(0,las))
a = np.greater(las, 0, dtype=np.float32)

yPred = [1.2, 1.4, 2, 0.9, 4, 0.0, -1., 2.3]
yPred = np.array(yPred)
pos = np.argwhere(np.abs(yPred-1) < 0.5)
a = 2.2
print(f'P: {a}')