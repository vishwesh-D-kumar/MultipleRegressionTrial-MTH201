import pandas as p
# import seaborn as s
import matplotlib.pyplot as plt
# %matplotlib inline
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def correl(x,y):
	n=len(x)
	sxy=0
	sx=0
	sy=0
	sx2=0
	sy2=0

	for i in range(n):
		sx+=x[i]
		sy+=y[i]
		sx2+=x[i]**2
		sy2+=y[i]**2
		sxy+=x[i]*y[i]

	r=((n*(sxy))-(sx*sy))/((((n*(sx2))-((sx)**2))*(((n*(sy2))-((sy)**2))))**0.5)
	sdx=((sx2/n)-((sx/n)**2))**0.5
	sdy=((sy2/n)-((sy/n)**2))**0.5
	regression=r*(sdy/sdx)

	return r,regression,sx/n,sy/n

def meansqerr(x,y):
	
	ans=0
	for i in range(len(x)):
		ans+=(x[i]-y[i])**2
	ans=ans/len(x)
	return ans





# data=p.read_csv(filepath_or_buffer="data1.csv",delimiter=";")
# data=data[["Title","H index"]]
# print(data)
# print(data.columns)
# z=""
data2=p.read_csv("found copy.csv",delimiter=",")
data4=p.read_csv("conference.csv",delimiter=";")
# print(data4.columns)
# data2=data2[["Title","ImpactFactor"]]
# for i in data2["Title"]:
# 	if i  in data1["Title"]
# 		z=z+i+";"
# # print(data2)
# print(data2.loc["Journal of Statistical Software"])
# data3=p.merge(data2,data,on="Title")
data3=data2
# print(data3["H index"].values)

# print(data3.columns.values)
# print("Correlation coeffficient is:",data3["ImpactFactor"].corr(data3["H index"]))
# print(data3)
# correl1=data3["ImpactFactor"].corr(data3["H index"])
# correl2=data3["H index"].corr(data3["ImpactFactor"])

# print(correl1)
# s.pairplot(x_vars='H index',y_vars='ImpactFactor',data=data3) 
# s.lmplot(x='H index',y='ImpactFactor',data=data3) 

# plt.show()
# c=0
# for i in data2["Title"].values:
# 	if i not in data3["Title"].values:
# 		print(i)
# 		c+=1
# print(len(data2["Title"]),len(data3["Title"]),c)
# train=data3.head(473)
# # print(train)

# predict=data3.tail(592-473)
# correl2=train["ImpactFactor"].corr(train["H index"])
# ym=train["ImpactFactor"].mean()
# xm=train["H index"].mean()
# print(xm,ym)
# y=(x-xm)correl2+
# s.lmplot(x='H index',y='ImpactFactor',data=train)

train_index,test_index,train_fact,test_fact = train_test_split(data3["H index"],data3["ImpactFactor"], test_size=0.2)
# print(train_index,test_index,train_fact,test_fact)
# print(train_index)
# print(train_fact)
l=correl(data3["H index"],data3["ImpactFactor"])
x=correl(train_index.values,train_fact.values)
crl=x[0]
m=x[1]
print("islope is ",m)
xm=x[2]
ym=x[3]
print("intercept ",ym-(m*xm))
print("Correlation coeffficient of H index and ImpactFactor over 100% of the data",l[0])
# y=m(x-xm)+ym
predictfact=[]
for i in range(len(test_index)):
	predictfact.append(m*(test_index.values[i]-xm)+ym)

sqerr=meansqerr(test_fact.values,predictfact)
print("Mean square error for Journal",sqerr)
# plt.scatter(tests_index.values,predictfact)

plt.scatter(test_index.values,test_fact.values)
t=list(range(200))
print("Regression line formed by training data is y=",m,"*","(","x-",xm,")","+",ym)

plt.plot(t,m*(t-xm)+ym,color="green",Label="Regression line")
# plt.show(block=False)
plt.pause(4)
# plt.close()
# predictconffact=[]
# for i in range(data4["H index"]):
# 	predictconffact.append(m*(i-xm)+ym)

data4["ImpactFactor"]=m*(data4["H index"]-xm)+ym
JournalTesting=p.DataFrame( columns=[["H index values","Impact Factor Predicted","Impact factor observed"]])
JournalTesting["H index values"]=test_index
JournalTesting["Impact Factor Predicted"]=predictfact
JournalTesting["Impact factor observed"]=test_fact
JournalTesting.to_csv("training_data-Predicted and expected.csv",index=False)

data4.to_csv("conferences_with_ImpactFactor.csv",index=False)




# train_index=train_index.values.reshape(-1, 1)
# train_fact=train_fact.values.reshape(-1,1)
# test_index=test_index.values.reshape(-1,1)
# # print(train_index,train_fact)
# l=LinearRegression()

# l.fit(train_index,train_fact)
# # print(l.coef_)
# # print(l.score)
# expected=l.predict(test_index)
# expectedconf=l.predict(data4["H index"].values.reshape(-1,1))
# z=[]
# q=[]
# for i in expected:
# 	z.append(i[0])
# for i in expectedconf:
# 	q.append(i[0])

# # print(z)

# sqerr=mean_squared_error(z,test_fact)
# print("Mean square error is :",sqerr)
# # plt.scatter(test_index,test_fact)
# # plt.scatter(test_index,z)
# plt.scatter(data4["H index"].values.reshape(-1,1),q)




# # fig, ax1 = plt.subplots()

# # color = 'tab:red'
# # ax1.set_xlabel('index')
# # ax1.set_ylabel('expected', color=color)
# # ax1.plot(test_index, expected, color=color)
# # # ax1.tick_params(axis='y', labelcolor=color)

# # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# # color = 'tab:blue'
# # ax2.set_ylabel('actual', color=color)  # we already handled the x-label with ax1
# # ax2.plot(test_index, test_fact, color=color)
# # ax2.tick_params(axis='y', labelcolor=color)
# # fig.tight_layout()

# plt.show()
# m=correl2
# for i in 


