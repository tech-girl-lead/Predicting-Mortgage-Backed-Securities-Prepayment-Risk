from django.shortcuts import render
from django.http import HttpResponse

import joblib
from joblib import load
model_path1 = "/Users/SalmaDkier/Desktop/TechnoDeployment/savedModels/model.joblib"
model_path2 = "/Users/SalmaDkier/Desktop/TechnoDeployment/savedModels/pca.joblib"
model_path3 = "/Users/SalmaDkier/Desktop/TechnoDeployment/savedModels/scaler.joblib"



model = joblib.load(model_path1)
pca_model = joblib.load(model_path2)
scaler = joblib.load(model_path3)




def predictor(request):
    return render(request,"page.html")

def form_info(request):
    CreditScore=request.GET['cs']
    MIP=request.GET['mip']
    Units=request.GET['units']
    OCLTV=request.GET['ocltv']
    DTI=request.GET['dti']
    OrigUPB=request.GET['origupb']
    LTV=request.GET['ltv']
    OrginIntRate=request.GET['oir']
    OrginLoanTerm=request.GET['olt']
    LoanPurposeC=request.GET['lpc']
    LoanPurposeN=request.GET['lpn']
    LoanPurposeP=request.GET['lpp']
    IsFirstTime =request.GET['firsttime']
    LTVRange=request.GET['ltvrange']
    CreditRange_Excellent=request.GET['cre']
    CreditRange_Fair=request.GET['crf']
    CreditRange_good=request.GET['crg']
    CreditRange_poor=request.GET['crp']
    RepayRange_H=request.GET['rrh']

    inputs=[CreditScore,MIP,Units,OCLTV,DTI,OrigUPB,LTV,OrginIntRate,OrginLoanTerm,LoanPurposeC,LoanPurposeN,LoanPurposeP,IsFirstTime,LTVRange,CreditRange_Excellent,CreditRange_Fair,CreditRange_good,CreditRange_poor,RepayRange_H]
    x= scaler.fit_transform([inputs])
    x_test=pca_model.transform(x)

    y_pred =model.predict(x_test)
    print(y_pred)
    if y_pred[0]==1:
        y_pred='Ever Delinquent'
    else :
        y_pred='Not Ever Delinquent'
    return render(request,'result.html',{'result':y_pred})