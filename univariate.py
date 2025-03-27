import pandas as pd
import numpy as np
import seaborn as sns
class Univariate():
    def quanQual(dataset):
        quan=[]
        qual=[]
        for columnName in dataset.columns:
            if(dataset[columnName].dtypes=='O'):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan,qual
    
    def MMM(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
        return descriptive  
    
    def Percentile(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"]
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"]
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"]
            descriptive.loc["99%",columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"]
        return descriptive    

    def IQR(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5rule","min","max","Lesser","Greater"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"]
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"]
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"]
            descriptive.loc["99%",columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"]
            descriptive.loc["IQR",columnName]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive.loc["1.5rule",columnName]=1.5*descriptive[columnName]["IQR"]
            descriptive.loc["Lesser",columnName]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5rule"]
            descriptive.loc["Greater",columnName]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5rule"]
            descriptive.loc["min",columnName]=dataset[columnName].min()
            descriptive.loc["max",columnName]=dataset[columnName].max()
        return descriptive    

    def FindingOutliers(descriptive,quan):
        lesser=[]
        greater=[]
        for columnName in quan:
            if (descriptive.loc["min"][columnName]<descriptive.loc["Lesser"][columnName]):
                lesser.append(columnName)
            if (descriptive.loc["max"][columnName]>descriptive.loc["Greater"][columnName]):
                greater.append(columnName)
        return greater,lesser 

    def ReplaceOutliers (lesser,greater,descriptive,dataset):
        for columnName in lesser:
            dataset[columnName][dataset[columnName]<descriptive[columnName]["Lesser"]]=descriptive[columnName]["Lesser"]
        for columnName in greater:
            dataset[columnName][dataset[columnName]>descriptive[columnName]["Greater"]]=descriptive[columnName]["Greater"]
        return dataset 

    def freqTable(dataset,quan):
        for columnName in quan:
            freqTable=pd.DataFrame(columns=["Unique_values","Frequency","Relative_Frequency","CumSum"])
            freqTable["Unique_values"]=dataset[columnName].value_counts().index
            freqTable["Frequency"]=dataset[columnName].value_counts().values
            freqTable["Relative_Frequency"]=(freqTable["Frequency"]/103)
            freqTable["CumSum"]=freqTable["Relative_Frequency"].cumsum()
        return freqTable

    def SkewKurt(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5rule","min","max","Lesser","Greater","Skewness","Kurtosis"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"]
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"]
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"]
            descriptive.loc["99%",columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"]
            descriptive.loc["IQR",columnName]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive.loc["1.5rule",columnName]=1.5*descriptive[columnName]["IQR"]
            descriptive.loc["Lesser",columnName]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5rule"]
            descriptive.loc["Greater",columnName]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5rule"]
            descriptive.loc["min",columnName]=dataset[columnName].min()
            descriptive.loc["max",columnName]=dataset[columnName].max()
            descriptive.loc["Skewness",columnName]=dataset[columnName].skew()
            descriptive.loc["Kurtosis",columnName]=dataset[columnName].kurtosis()
        return descriptive  

    def VAR_SD(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5rule","min","max","Lesser","Greater","Variance","SD"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"]
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"]
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"]
            descriptive.loc["99%",columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"]
            descriptive.loc["IQR",columnName]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive.loc["1.5rule",columnName]=1.5*descriptive[columnName]["IQR"]
            descriptive.loc["Lesser",columnName]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5rule"]
            descriptive.loc["Greater",columnName]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5rule"]
            descriptive.loc["min",columnName]=dataset[columnName].min()
            descriptive.loc["max",columnName]=dataset[columnName].max()
            descriptive.loc["Variance",columnName]=dataset[columnName].var()
            descriptive.loc["SD",columnName]=dataset[columnName].std()
        return descriptive    

    def get_pdf_probabilty(dataset,startrange,endrange):
        from matplotlib import pyplot
        from scipy.stats import norm
        ax=sns.distplot(dataset,kde=True,kde_kws={"color":"blue"},color='Green')
        pyplot.axvline(startrange,color='Red')
        pyplot.axvline(endrange,color='Red')
        #generate a sample
        sample=dataset
        #calculate parameters
        sample_mean=sample.mean()
        sample_std=sample.std()
        print("Mean=%.3f,Standard Deviation=%.3f"%(sample_mean,sample_std))
        #define the distribution
        dist=norm(sample_mean,sample_std)
        #sample probabilities for a range of outcomes
        values=[value for value in range(startrange,endrange)]
        probabilities=[dist.pdf(value) for value in values]
        prob=sum(probabilities)
        print("The area between range({},{}):{}".format(startrange,endrange,sum(probabilities)))
        return prob

    def compute_ecdf(dataset,columnName,value):
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf=ECDF(dataset[columnName])
        return ecdf(value)

    def StdNBgraph(dataset):
        #converted to Standard Normal Distribution
        mean=dataset.mean()
        std=dataset.std()
        values=[i for i in dataset]
        z_score=[((j-mean)/std)for j in values]
        sns.displot(z_score,kde=True)
        sum(z_score)/len(z_score)
                    