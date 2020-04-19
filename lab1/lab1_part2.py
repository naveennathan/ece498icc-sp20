import numpy as np
#Part 2: Numpy Basics
A = np.zeros( (9, 6) )
print("A")
print(A)
A[:, 2:4] = 1
A[0:2] = [0, 1, 1, 1, 1, 0]
A[7:9] = [0, 1, 1, 1, 1, 0]
print("A")
print(A)
B = np.vstack([ np.zeros(6), A, np.zeros(6) ])
print("B")
print(B)
C = np.arange(1, 67).reshape(11, 6)
print("C")
print(C)
D = B * C
print("D")
print(D)
E = D[ D != 0 ]
print("E")
print(E)
max, min = E.max(), E.min()
F = (E - min) / (max - min)
print("F")
print(F)
print(F[(np.abs( F - 0.25 )).argmin()])
