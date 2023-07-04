import numpy as np
import pandas as pd
import pickle

model=pickle.load(open('./Models/LSTM_model.pkl','rb'))
scaler=pickle.load(open('./Models/scaler_model.pkl','rb'))
df=pd.read_csv('.\Data\ice.csv')
df1=df.reset_index()['Icecream']
df1=df1[527:]


def predict(steps):
    x_input=scaler.transform(np.array(df1).reshape(-1,1))
    x_input=x_input.reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=50
    i=0
    while(i<steps):
        
        if(len(temp_input)>50):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    k=scaler.inverse_transform(lst_output)
    k=k.reshape(1,-1)
    k=list(k)
    k=k[0].tolist()
    return k
