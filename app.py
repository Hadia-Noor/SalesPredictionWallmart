from flask import Flask,render_template,request

app = Flask(__name__)
import pickle
import numpy as np
import pandas as pd

RFRegressor=pickle.load(open('store.pkl' ,'rb')) 
@app.route('/')
def form():
    return render_template('index.html') 
@app.route('/back/', methods=['POST'])
def back():
    return render_template('index.html') 
@app.route('/prediction/', methods = ['POST'])
def prediction():
        if request.method=='POST':
            Store=request.form.get("store")
            Dept=request.form.get("dept")
            IsHoliday=request.form.get("holiday")
            Size=request.form.get("size")
            Week=request.form.get("week")
            Type=request.form.get("type")
            Year=request.form.get("year")
            if IsHoliday=='False':
                IsHoliday = 0;
            if IsHoliday == 'True':
                IsHoliday=1;       

            cols = ["Store","Dept", "IsHoliday","Size","Week", "Type","Year"]
            features = [Store,Dept, IsHoliday,Size,Week, Type,Year]
            final_features = [features]
            df=pd.DataFrame(final_features,columns=cols)
            predictions= RFRegressor.predict(df)
            return render_template('output.html', value=(str(predictions[0])))
        
        

    
if __name__ == "__main__":
  app.debug  
  app.run()