import numpy as np
import pandas as pd
import xlrd
import xlwt
import openpyxl

def three_sigma(Ser1):
    '''
    Ser1：DataFrame
    '''
    rule = (Ser1.mean()-3*Ser1.std()>Ser1) | (Ser1.mean()+3*Ser1.std()< Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    outrange = Ser1.iloc[index]
    return outrange


def readxl(xlfile,num):
    with pd.ExcelWriter(xlfile+'.3sigma.xlsx') as writer:
        workBook = xlrd.open_workbook(xlfile)
        allSheetNames = workBook.sheet_names()
        for i in allSheetNames:
            df2=pd.read_excel(xlfile,sheet_name=i)
            for col in range(int(num)-1,len(df2.columns.tolist())):
                colname=df2.columns.tolist()[col]
                outrage=three_sigma(df2[colname]).index.tolist()
                for x in outrage:
                    df2.loc[x,colname]='NA'
            df2.to_excel(writer,sheet_name=i,index=False)

readxl(files,num)