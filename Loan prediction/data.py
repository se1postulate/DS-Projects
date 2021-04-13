# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:22:05 2021

@author: AMK
"""

import pandas as pd
import numpy as np
data=pd.read_csv("Data/data.csv")

def encode(data):
    datagender=data.Gender.unique()
    datamarried=data.Married.unique()
    dataproperty=data.Property_Area.unique()
    dataedu=data.Education.unique()
    dataemp=data.Self_Employed.unique()
    dataloan=data.Loan_Status.unique()
    Gender=data.Gender.replace(datagender,np.arange(len(datagender)))
    Married=data.Married.replace(datamarried,np.arange(len(datamarried)))
    Property_Area=data.Property_Area.replace(dataproperty,np.arange(len(dataproperty)))
    Education=data.Education.replace(dataedu,np.arange(len(dataedu)))
    Loan_Status=data.Loan_Status.replace(dataloan,np.arange(len(dataloan)))
    Self_Employed=data.Self_Employed.replace(dataemp,np.arange(len(dataemp)))
    Dependents=data.Dependents.replace('3+',3)
    ApplicantIncome=data.ApplicantIncome
    CoapplicantIncome=data.CoapplicantIncome
    LoanAmount=data.LoanAmount
    Loan_Amount_Term=data.Loan_Amount_Term
    Credit_History=data.Credit_History
    df={'Gender':Gender,'Married':Married,'Dependents':Dependents,'Education':Education,'Self_Employed':Self_Employed,'ApplicantIncome':ApplicantIncome,'CoapplicantIncome':CoapplicantIncome,'LoanAmount':LoanAmount,'Loan_Amount_Term':Loan_Amount_Term,'Credit_History':Credit_History,'Property_Area':Property_Area,'Loan_Status':Loan_Status}
    data=pd.DataFrame(df)
    return data