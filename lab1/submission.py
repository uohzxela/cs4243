import numpy as np
import numpy.linalg as la

file = open("data.txt")
data = np.genfromtxt(file, delimiter=",") 
file.close()
print "data ="
print data

M = []
b = []

for x_prime, y_prime, x, y in data:
	M.append([x,y,1,0,0,0])
	M.append([0,0,0,x,y,1])
	b.append([x_prime])
	b.append([y_prime])

M = np.matrix(M)
print "M ="
print M
b = np.matrix(b)
print "b ="
print b

a, e, r, s = la.lstsq(M, b)
print "a ="
print a

# print "M*a =", M*a
# print "b =", b

sum_squared_error = pow(la.norm(M*a-b), 2)
print "Sum-squared error =", sum_squared_error
print "Residue =", e