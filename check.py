import pandas as pnd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import difflib
import numpy as np


scimajor=pnd.read_csv("scimagojr.csv" ,delimiter=";")
found=pnd.read_csv("found copy.csv",delimiter=",")
# print(scimajor.columns)
from itertools import *

def abserr(x,y):
	s=0
	for i in range(len(x)):
		s+=abs(x[i]-y[i])
	s=s/len(x)

	return s

def predictor(train_index,train_fact,test_index):
	x=[]
	for i in list(train_index.columns):
		x.append(list(train_index[i].values))
	# print(x)


	b=multipleregression(x,list(train_index.values))
	# l=b.tolist()

	# # for i in l:
	# # 	i.append(1)
	# b=np.array(l)
	# print(l)

	x=np.array(test_index).T

	
	# print(x)
	print(x.shape,b.shape)

	return np.dot(b,x).tolist()



def multipleregression(a,y):
	# a=[[1,2],[3,4]]
	# y=[[2,3],[3,4]]

	x=np.array(a)
	# print(x)
	# print(a)
	y=np.array(y).T
	
	
	xt=x.T
	# print(xt)
	# return
	# print(np.linalg.det(np.dot(xt,x)))
	# print(x.tolist(),xt.tolist())

	# print()
	# return
	# return
	# inv=


	return np.dot(np.dot(np.linalg.inv(np.dot(xt,x)),xt),y.T)

def strcheck(x):
	if x.dtype=="object":
		# print("Skipping")
		return True
	else:
		# print("Good to go")
		return False

def subgen(n):
	l=[]
	x=[]
	y=[]

	for i in range(1,n+1):
		x.append(0)
		l.append(list(combinations(list(range(n)),i)))

	# print(l)

	# for i in l:
		
	# 	for k in i:
	# 		z=x[:]
	# 		for j in k:
	# 			z[j]=1
	# 		y.append(z)
		
	return(l)



def meansqerr(x,y):
	
	ans=0
	for i in range(len(x)):
		ans+=(x[i]-y[i])**2
	ans=ans/len(x)
	return ans

# Removing spaces and commas and dashes from title to go aheqd and match perfectly with found.csv

# print(scimajor["Title"])




scimajor["Title"]=scimajor["Title"].str.replace(",","")

found["Title"]=found["Title"].str.replace(",","")
scimajor["Title"]=scimajor["Title"].str.replace(":","")
found["Title"]=found["Title"].str.replace(":","")

scimajor["Title"]=scimajor["Title"].str.replace("-"," ")
found["Title"]=found["Title"].str.replace("-"," ")
scimajor["Title"]=scimajor["Title"].str.replace(" ","")
found["Title"]=found["Title"].str.replace(" ","")


# found["Title"]=found["Title"].str.lower()
# scimajor["Title"]=scimajor["Title"].str.lower()

# print(scimajor["Title"])




#merging dataframes on basis of their title
new=pnd.merge(scimajor,found, on="Title")
# print(new.columns)
# l=LinearRegression()



# print(new["SJR"])




# Cleaning the SJR,and other files indexes of the commas, replacing them with .
for j in ["Cites / Doc. (2years)","Ref. / Doc.","SJR"]:
	baka=[]
	for i in new[j]:
		# print(type(i),i)
		if (type(i)==float and math.isnan(i)):
			baka.append(1)
			continue
		baka.append(float(str(i).replace(",",".")))

	new[j]=baka




# # print(new["SJR"],new["H index_x"])
# l.fit(new["SJR"].values.reshape(-1,1),new['ImpactFactor'].values.reshape(-1,1))
# # plt.plot(new["SJR"]/1000,new['ImpactFactor'])
# plt.show()
# print(l.score(new["SJR"].values.reshape(-1,1),new['ImpactFactor'].values.reshape(-1,1)))

# script to check if any unmatched journals remain
c=0
print(len(found),len(new))
for i in found["Title"].values:
	if i not in new["Title"].values :
		c+=1
		
		print(i)

for i in new["Title"].values:
	if i.find("cybernetics")>0:
		print(i+"SAME")


print("Unmatched journals are :",c)###########################IMPORTANT###########
 

cols=list(new.columns)
# print(new["Cites / Doc. (2years)"])
# print(cols)
##Removing the string types
i=0
while (True):
	
	if i==len(cols):
		break

	if strcheck(new[cols[i]]):
		cols.pop(i)
	else:
		i+=1
cols.pop(-2)
cols.pop(0)
# cols.pop(-2)
cols.pop(0)
cols.pop(0)
cols.pop(-1)
print("Columns to be considered ",cols)
# print(new["Publisher"].dtype)
# for i in cols:
	# print(new[i].dtype)

# new.to_csv("new.csv",index=False)


###Generating all subsets for columns , except Impact Factor 
subsets=subgen(len(cols))



# print(len(subsets),len(cols))
# print(subgen(4))



#for string data types :assigning a number ot each diff string ie; Elsevier BV==> 8,SAGE Publications=>1


def colname(l):
	x=""
	for i in l:
		x=x+","+cols[i]
	return x



#iterating over all subsets
errors={}
abserrors={}
for i in subsets:
	for j in i:
		x=pnd.DataFrame()
		# print(i,j,"This is ")
		for k in j:

			if strcheck(new[cols[k]]):
				l=set(new[cols[k]].values)
				strdic={}
				b=0
				for z in l:
					strdic[z]=b
					b+=1

				l=[]
				for z in new[cols[k]].values:
					l.append(strdic[z])
				x[cols[k]]=l




	
			else:
				x[cols[k]]=new[cols[k]]
			# print(x)
			y=new["ImpactFactor"]
		train_index,test_index,train_fact,test_fact = train_test_split(x,y ,test_size=0.2,random_state=101)
		# print(colname(j))
		# print(train_index)
	
####Checking for singular matrices ,if there using sklearn
		try:

			predicted=predictor(train_index,train_fact,test_index)
		# if i ==subsets[1] and j==subsets[1][1]:
		# 	print(colname(j))

		# 	break

		except:
			l=LinearRegression()
			l.fit(train_index,train_fact)
			predicted=l.predict(test_index)

		errors[colname(j)]=meansqerr(predicted,test_fact.values)
		abserrors[colname(j)]=abserr(predicted,test_fact.values)

# print(errors,"Yes")
leasterr=min(list(errors.values()))

print(min(list(errors.values())),"Min mean sqrrors")



for i in errors:
	if errors[i]==leasterr:
		print(i[1:])
		break
print(abserrors)
print(errors)


print(min(list(abserrors.values())),"Min mean abserrors")

leasterr=min(list(abserrors.values()))
# print(leasterr,"meanerr")
abserrorscsv=pnd.DataFrame()
abserrorscsv["Combo"]=[i[1:] for i in abserrors]
abserrorscsv["Abserrors"]=abserrors.values()
abserrorscsv.to_csv("AbsErrorsOfCombos.csv",index=False)

meansqerrcsv=pnd.DataFrame()
meansqerrcsv["Combo"]=[i[1:] for i in errors]
meansqerrcsv["Abserrors"]=errors.values()
meansqerrcsv.to_csv("MeanSquareErrorsOfCombos.csv",index=False)
for i in abserrors:

	if abserrors[i]==leasterr:
		print(i[1:])
		break







