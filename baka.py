from random import randint as randi
# from numpy.linalg import inv
import numpy as np
def baka(x):

	for i in range(len(x)):
		for j in x[i]:
			print(j,end=" ")
		print()
def matrixInverter(matr):
	det=determinant(matr,len(matr))
	adj=adjoint(matr)
	# adj=transpose(adj)
	print(baka(adj))
	for i in range(len(adj)):
		for j in range(len(adj)):
		
			adj[i][j]=adj[i][j]/det
	return adj

def adjoint(matr):
	x=[]
	s=1
	for i in range(len(matr)):
		x.append([])
		
		for j in range(len(matr)):
			x[i].append(s*determinant(cofact(matr,i,j),len(matr)-1))
			s=-s

		s=-s
	x=transpose(x)
	return x



def transpose(matr):
	# print(matr)
	y=matr
	# for i in range(len(matr)):
	# 	x.append([])
	# 	for j in range(len(matr)):
	# 		y[i].append(0)


	# print(y,"gjhgj")
	for i in range(len(matr)):
		# x.append([])
		for j in range(len(matr)):
			y[i][j]=matr[j][i]
	return y


def determinant(matr,n):
	# print(len(matr))
	if n==1:
		return(matr[0][0])
	d=0
	s=1
	print (n)
	for i in range(len(matr)):
		d+=s*matr[i][0]*determinant(cofact(matr,i,0),n-1)
		# print(determinant(cofact(matr,i,0),n-1))
		s=-s
	return d
def cofact(matr,l,m):
	print(l,m)
	cfact=[]
	for i in range(len(matr)-1):
		cfact.append([])
		for j in range(len(matr)-1):

			cfact[i].append(0)
	i=0
	j=0

	for a in range(len(matr)):
		# print("YEsh")
		if a==l:
			continue
		
		for b in  range(len(matr)):
			# print(a,b,i,j)
			# print(matr,i,j,len(matr))
			if (b==m):
				continue
			else:

				cfact[i][j]=matr[a][b]
				j+=1

		j=0	
		i+=1

	return(cfact)

x=[]
for i in range(620):
	x.append([])
	for j in range(620):
		x[i].append(randi(1,10))
# x=[[22,0,1,3],[4,6,8,9],[4,3,6,1],[2,5,1,8]]
# print(baka(x))
# print(matrixInverter(x))

# print(inv(x))


# print(transpose(x))
a=[[1],[3]]
# y=[[2,3],[3,4]]
# print(np.array(a).T.tolist())




