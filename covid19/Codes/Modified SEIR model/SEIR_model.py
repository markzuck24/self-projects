# -*- coding
import os
import requests
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import date,timedelta

# Change Directory to save the Output files in the desired path
# Edit Working Directory
working_directory = 'C:\\Users\\prabhu.muddada\\Documents\\COVID\\Epidemic Model SEIR V10'
os.chdir(working_directory)

select_prediction_level = ["Country","State","City"] #"Country" / "State" / "City"

reqd_city_df_orig = pd.read_csv('Hotspot_regions.csv')

###############################################################################

# Get raw data from API

def get_data_from_api(url):
    """
    Function to get data from api
    
    Takes url of the json as input. Returns a dictionary of pandas dataframes.
    """
    print("\n")
    print(url)
    req_json = requests.get(url).json()
    data_dict = {k: pd.DataFrame(v) for k, v in req_json.items()}
    print(data_dict.keys())
    return data_dict


# Data - Getting Raw Data from covid19india.org
url_raw_data = "https://api.covid19india.org/raw_data.json"
d_raw_data = get_data_from_api(url_raw_data)

# Converting Data into a Dataframe from Dictionary
raw_df = d_raw_data.get("raw_data")

#Removing Null records
raw_df = raw_df[~raw_df['dateannounced'].isin([''])]

# Renaming Thane to Mumbai, as Thane comes under Mumbai.
raw_df['detecteddistrict'][raw_df['detecteddistrict']=='Thane'] = 'Mumbai'
# Renaming 'Bengaluru Rural' to 'Bengaluru'
raw_df['detecteddistrict'][raw_df['detecteddistrict']=='Bengaluru Rural'] = 'Bengaluru'


#%%
###############################################################################

def after_min_cases(df,min_case):
    '''
    function to return cases after specified minimum cases
    '''
    df1 = df[df['Cases'] >= min_case]
    df1['Days'] = 0
    df1['Days'] = df1['Date'].apply(lambda x: (x - df1['Date'].iloc[0]).days)
    return (df1)

def basicplots(df,min_case):    
    plt.plot(df['Date'], df['Cases'])
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.xlabel('Date')
    plt.grid(True)
    plt.show()
    
    df1 = after_min_cases(df,min_case)
    plt.plot(df1['Days'], df1['Cases'])
    plt.yscale("log")
    plt.xlabel('Days after 50th case')
    plt.grid(True)
    plt.show()
    
    return(df1)

#Extension to SEIR model
# The SEIR model differential equations.
def derivseir(y, t, N, beta, alpha, gamma, lockdown_fraction=0, quarantine_fraction=0):
    '''
    Function to compute the rate of change for various compartments
    
    This model has a lockdown fraction - of all the suscepted, a certain fraction are in lockdown, while 
    others escape lockdown and might get exposed to disease.
    
    # If S are suscepted, [(1-lockdown_fraction) X S] will be roaming free, and got a chance to infect
    # If lockdown_fraction == 1, all suscepted are in lockdown, and disease will not spread to them.
    '''
    
    S, E, I, R = y
    
    # fraction of all suscepted, who are not in lockdown    
    newS = (1-lockdown_fraction) * S
    
    # fraction of all suscepted, who are not in lockdown    
    newE = (1-quarantine_fraction) * E 
    
    # new rate of change for susceptible people - based on those that escape lockdown
    #dSdt = -beta * newS * I / N #previous equation
    dSdt = -beta * newS * newE / N  #modified equation 
    
    #new rate of change for exposed people
    #dEdt = beta * newS * I / N - alpha * E #previous equation
    dEdt = beta * newS * newE / N - alpha * newE  #modified equation 
    
    #this will not change, as those already exposed will start showing symptoms
    dIdt = alpha * newE - gamma * I 
    
    #this will not change, as all those infected will recover or reach an outcome
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt



def predictcases(population, infected, incubation_period, infectious_period, beta, \
                 EbyIratio=2.4, days=30,  lockdown_fraction=0, quarantine_fraction=0): 
    '''
    function to predict the number of cases, based on population, initial infected and parameters
    Input parameters
    population: population of city/state/country
    infected: no of infected cases at start
    incubation_period: time to show symptoms (in days)
    infectious_period: time duration for which a patient can infect others (in days)
    beta: sent in as a value. this is being estimated by the meta function that calls this function
    EbyIratio: exposed to infected ratio
    
    '''
    ###########################################################################
    ## OLD VALUES - NOW BEING PASSED FROM OUTSIDE THIS FUNCTION
    ## Parameters from literature
    
    ### Incubation period 
    ### (https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported)
    #incubation_period = 5 # days
    
    ## Infectious period
    ## (https://www.who.int/bulletin/online_first/20-255695.pdf)
    #infectious_period  = 2.3 #days
    
    ## Exposed to infected ratio
    ## (https://www.who.int/bulletin/online_first/20-255695.pdf)
    #ebyiratio = 2.4        
    ###########################################################################
    alpha = 1/incubation_period
    #print (alpha)
    
    gamma = 1/infectious_period
    #print (gamma)
    
    # Total population, N.
    
    # Initial number of infected, exposed and recovered individuals, I0 and R0.
    I0 = infected
    E0 = EbyIratio  * I0
    R0 = 0
    N = population
    
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - E0 - I0 - R0
    
    # A grid of time points (in days)
    t = np.linspace(0, days, days+1)
    #print (t)

    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SEIR equations over the time grid, t.
    ret = odeint(derivseir, y0, t, args=(N, beta, alpha, gamma, lockdown_fraction,quarantine_fraction)) 
    S, E, I, R = ret.T
    
    return (t, S, E, I, R)


def esterror(actual_cases_train,actual_cases_test, t, S, E, I, R,overpredict_percent):
    '''
    function to estimate the RMS error between true and predicted values
    actual_cases_train: array of daily cases used for Training
    actual_cases_train: array of daily cases used for Testing
    t: time
    S: daily susceptible cases
    E: daily exposed cases
    I: daily infected cases
    R: daily recovered cases
    '''
    # true cases
    y_true = np.array(actual_cases_train) 
    
    # predicted cases - both infected as well as recovered
    y_pred = np.array(I[:len(y_true)] + R[:len(y_true)])
    
    diff = (y_true - y_pred)/(y_true)
    
    # Calculating Error in Train Data
    error = np.sqrt(np.mean(diff**2))
    
    y_true_test = np.array(actual_cases_test) 
    
    y_pred_test = np.array(I[len(actual_cases_train):] + R[len(actual_cases_train):])
    
    diff_test = (y_true_test - y_pred_test)/(y_true_test)
    
    y_true_test = y_true_test * (1 + overpredict_percent/100)
    
    diff_test_percent = (y_true_test - y_pred_test)/(y_true_test)
    
    # Flag to check whether all Predictions on Test Data are greater than atual Test Data by 5%.
    # This is to make sure the model is over predicting by more than or eqal to 5%.
    overpredict_test_by_percent_flag = all(i <= 0 for i in diff_test_percent)
    
    # Flag to check whether all Predictions on Test Data are greater than atual Test Data.
    # This is to make sure the model is not under predicting.
    overpredict_test_flag = all(i <= 0 for i in diff_test)
    
    return (error,overpredict_test_by_percent_flag,overpredict_test_flag)


## function to estimate best beta and return the best value
## strategy: check in larger intervals first, eg. 1, 1.2, 1.4, 1.6, 1.8, 2.0
## if beta = 1.2, then, check in smaller intervals, eg. 1.1, 1.12, 1.14, 1.16, ..1.20, 1.22, 1.24, ...1.30.
def estbeta(df, pop, infected, days, betamax, betamin, incubation_period=5, infectious_period=2.3, 
            EbyIratio=2.4,  nint=21, lockdown_fraction=0,quarantinemin=0,quarantinemax=1, test_days=2, overpredict_percent=5): 
    '''
    function to estimate the best beta.
    step 1: Return a beta value which is overpredicting in test data by more than overpredict_percent & has minimum error.
            If none of the beta's is following above condition, go to step 2.
    step 2: Return a beta value which is overpredicting in test data between 0 to overpredict_percent & has minimum error.
            If none of the beta's is following above condition, go to step 3.
    step 3: Return a beta value which is underpredicting in test data & has minimum error.
    
    variables to predict cases function
    predictcases(population, infected, incubation_period, infectious_period, beta, EbyIratio=2.4, days=30):
    '''

    minerror_overpredict_percent = 1e5
    minerror_overpredict = 1e5
    minerror_underpredict = 1e5
    finalbeta_overpredict_percent = 0
    finalbeta_overpredict = 0
    finalbeta_underpredict = 0
    
    betalist = [round(x,2) for x in list(np.linspace(betamin, betamax, nint))]
    print (betalist)
    
    
    quarantinelist = [round(x,2) for x in list(np.linspace(quarantinemin, quarantinemax, 10))] 
    
    
    for quarantine_fraction in quarantinelist:

        for betal in betalist:
            #print (betal)
            
            t,S,E,I, R = predictcases(population= pop, infected = infected, days=days, \
                                      incubation_period=incubation_period, \
                                      infectious_period = infectious_period, beta = betal, EbyIratio=EbyIratio, 
                                      lockdown_fraction = lockdown_fraction,
                                      quarantine_fraction = quarantine_fraction) 
            
            actual_cases_train = np.array(df.iloc[range(len(df)-test_days),:]['Cases'])
            
            actual_cases_test = np.array(df.tail(test_days)['Cases'])
            
            error,overpredict_test_by_percent_flag,overpredict_test_flag = esterror(actual_cases_train,actual_cases_test, t, S, E, I, R,overpredict_percent)
            
           
            if ((error < minerror_overpredict_percent) & overpredict_test_by_percent_flag): 
                finalbeta_overpredict_percent = betal
                minerror_overpredict_percent = error
                quarantine_fraction_overpredict_percent = quarantine_fraction 
            
            
            if ((error < minerror_overpredict) & overpredict_test_flag): 
                finalbeta_overpredict = betal
                minerror_overpredict = error
                quarantine_fraction_overpredict = quarantine_fraction 
                
            if ((betal == betamax) &(quarantine_fraction == quarantinemin)):
                finalbeta_underpredict = betal
                minerror_underpredict = error
                quarantine_fraction_underpredict = quarantine_fraction 
        #print (finalbeta)
        
    if(finalbeta_overpredict_percent != 0):
        # RETURN the beta which is overpredicting in test data by more than overpredict_percent &  has minimum error
        return (finalbeta_overpredict_percent,minerror_overpredict_percent,quarantine_fraction_overpredict_percent) 
    elif(finalbeta_overpredict != 0):
        # RETURN the beta which is not overpredicting in test data by 0 to overpredict_percent &  has minimum error
        return (finalbeta_overpredict,minerror_overpredict,quarantine_fraction_overpredict)
    else:
        # RETURN the betamax and quarantinemin values if we fail to find overpredict cases. 
        return (finalbeta_underpredict,minerror_underpredict,quarantine_fraction_underpredict) 
           
    
def impute_missing_dates(df):
    '''
    Function to impute the missing dates
    '''
    S_date = df.Date.iloc[0]
    E_date = date.today() - timedelta(days=1)
    dates = pd.DataFrame([S_date + timedelta(days=i) for i in range((E_date - S_date).days + 1)],columns = ['Date'])
    df = dates.merge(df,how='left',on='Date')
    df = df.ffill()
    return(df)

    
def get_cummulative_count(df):
    '''
    Function to calculate cummulative count of cases.
    '''
    # Grouping By on Date
    df = pd.DataFrame(df.groupby(['dateannounced'])['dateannounced'].count())
    df.columns = ['count']
    df.reset_index(inplace=True)
    df.rename(columns = {'dateannounced':'Date'}, inplace = True)
    df = df.sort_values(['Date'])
    df['Cases'] = df['count'].cumsum()
    df.drop(['count'],inplace=True,axis=1)
    df['LogCases'] = df['Cases'].apply(lambda x: np.log(1+x))
    return(df)
    
    
def grid_search(df,citypop, cityinfected,days,lockdown_fraction_list,quarantinemin,quarantinemax,betamax,betamin 
                ,incubation_period=5.1, infectious_period=2.3,test_days=2,overpredict_percent=5,EbyIratio=2.4):
 
    '''
    Grid Search to find the best combination of lockdown_fraction & 
    beta value with minimal error.
    '''
    
    #lockdown_fraction_list = list(np.linspace(lockdownmin, lockdownmax, int((lockdownmax-lockdownmin)/0.1) + 1))
    #lockdown_fraction_list = list(np.linspace(lockdownmin, lockdownmax, int(round((lockdownmax-lockdownmin),1)/0.1) + 1))
    #lockdown_fraction_list = [round(x,2) for x in lockdown_fraction_list]
    
    params_df = pd.DataFrame(columns= ['incubation_period','infectious_period','lockdown_fraction','quarantine_fraction','best_beta','error']) 
    
    for lockdown_fraction in lockdown_fraction_list:
        #Checking beta in larger intervals
        bmax= betamax
        bmin = betamin
        nint = int((bmax-bmin)/0.01) + 1
        ## get the estimate of beta in larger intervals
        betaest,error,qf= estbeta(df, citypop, cityinfected, days,lockdown_fraction = lockdown_fraction,quarantinemin=quarantinemin,quarantinemax=quarantinemax, 
                                 betamax = bmax, betamin= bmin,incubation_period=incubation_period,
                                 infectious_period=infectious_period, nint=nint,test_days=test_days,overpredict_percent=overpredict_percent,EbyIratio=EbyIratio)
        
        params_df= params_df.append(pd.DataFrame([[incubation_period,infectious_period,lockdown_fraction,qf,betaest,error]],columns= ['incubation_period','infectious_period','lockdown_fraction','quarantine_fraction','best_beta','error']))

    
    params_df.sort_values(['lockdown_fraction'],ascending=True,inplace=True)
    params_df.reset_index(inplace=True,drop=True)
    
    
    return(params_df)
    
    
def plotgraph(df,cityname,days,min_case):
   
    predicted_cols = [s for s in df.columns if 'Predicted' in s]
   
    max_val1 = round((df['Actual'].max()) * 10, 0)
    max_val2 = round((df[predicted_cols].max(axis = 0, skipna = True)).max(), 0)
    if(max_val1 < max_val2):
        max_val = max_val1
    else:
        max_val = max_val2
        

    plt.figure(figsize=(10,10))

    plt.plot(df['Days'],df[predicted_cols])
    plt.plot(df['Days'],df[['Actual']],linestyle='-', marker='*',markersize=10,color='blue') 
   
    plt.ylim(0, max_val) 
    plt.xlim(0, days)
    legends_list = []
    for i in predicted_cols:
        lf_val = i[-3:]
        string = 'LF='+lf_val+',QF='+str(df['best_qf_at_lf_'+lf_val][0])+',Beta='+str(df['best_beta_at_lf_'+lf_val][0])+',Ro='+str(df['Ro_num_at_lf_'+lf_val][0]) 
        legends_list.append(string)
   
    plt.legend(legends_list + ['Actual'])
    plt.grid(True)
    plt.xlabel('Days since '+str(min_case) +'th case')
    plt.ylabel('Total # of cases')
    plt.title(cityname)
    plt.savefig(cityname+'_Results.png')
    plt.show()
    return ()


def plotgraph2(df,cityname,days,min_case):
   
    predicted_cols = [s for s in df.columns if 'Infected_per_day' in s]
   
    max_val =(df[predicted_cols].max(axis = 0, skipna = True)).max()
   
    plt.figure(figsize=(10,10))

    plt.plot(df['Days'],df[predicted_cols])
    plt.plot(df['Days'],df[['Actual_infected_per_day']],linestyle='-', marker='*',markersize=10,color='blue') 
   
    plt.ylim(0, max_val) 
    plt.xlim(0, days)
    legends_list = []
    for i in predicted_cols:
        lf_val = i[-3:]
        string = 'LF='+lf_val+',QF='+str(df['best_qf_at_lf_'+lf_val][0])+',Beta='+str(df['best_beta_at_lf_'+lf_val][0])+',Ro='+str(df['Ro_num_at_lf_'+lf_val][0]) 
        legends_list.append(string)
   
    plt.legend(legends_list + ['Actual_infected_per_day'])
    plt.grid(True)
    plt.xlabel('Days since '+str(min_case) +'th case')
    plt.ylabel('Total # of cases')
    plt.title(cityname+' - Infected per day')
    plt.savefig(cityname+'_InfectedPerDay.png')
    plt.show()
    return ()
   
    
def write_results(df,best_params_df,cityname,citypop,cityinfected,min_case,days=30,EbyIratio=2.4):
    '''
    Function to write results.
    To run SEIR model with each of the best pairs of beta and qauarantine fraction and save the predictions in an excel sheet 
    To save the plots of distribution of actual and predicted values. 
    '''
    #Changing Directory to output folder
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,cityname )
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
    os.chdir(final_directory)
    
    # Create final predicted dataframe
    # Assign constant values (not model results)
    predicted_df = pd.DataFrame(index = range(days + 1))
    predicted_df['City'] = cityname 
    predicted_df['CityPopulation'] = citypop
    predicted_df['Date'] = [df.Date.iloc[0] + timedelta(days=i) for i in range(len(predicted_df))]
    predicted_df = predicted_df.merge(df,how='left',on='Date')
    predicted_df['Days'] = range(len(predicted_df))
    predicted_df.rename(columns = {'Cases':'Actual'}, inplace = True)
    predicted_df['incubation_period'] = best_params_df['incubation_period'][0]
    predicted_df['infectious_period'] = best_params_df['infectious_period'][0]
    predicted_df = predicted_df[['City', 'CityPopulation', 'Date', 'Days','Actual', 'LogCases','incubation_period', 'infectious_period']]

    # Get predicted results for the best beta value
    for sol_no,params in best_params_df.iterrows():
    
        best_beta = round(params['best_beta'],2)
        
        lockdown_fraction = round(params['lockdown_fraction'],1)
        
        quarantine_fraction = round(params['quarantine_fraction'],1) 
        
        incubation_period = params['incubation_period']
        
        infectious_period = params['infectious_period']
        
        error = params['error']
        
        ## Estimate values of the best beta
        t,S,E,I, R = predictcases(population= citypop, infected = cityinfected, incubation_period=incubation_period, lockdown_fraction = lockdown_fraction, quarantine_fraction = quarantine_fraction,
                                  infectious_period = infectious_period, beta = best_beta, EbyIratio=EbyIratio, days=days)
        
        
        print("Best Beta:",best_beta)
        
        print("lockdown_fraction:",lockdown_fraction)
        
        predicted_df['best_beta_at_lf_'+str(lockdown_fraction)] = best_beta
        
        predicted_df['best_qf_at_lf_'+str(lockdown_fraction)] = quarantine_fraction 
        
        predicted_df['Ro_num_at_lf_'+str(lockdown_fraction)] = round((best_beta * incubation_period),2) 
        
        predicted_df['Predicted_at_lf_'+str(lockdown_fraction)] = (I+R).round()
        
        predicted_df['Error_at_lf_'+str(lockdown_fraction)] = error
        
        predicted_df['Infected_at_lf_'+str(lockdown_fraction)] = (I).round()
        
        predicted_df['Recovered_at_lf_'+str(lockdown_fraction)] = (R).round()
        
        predicted_df['Infected_per_day_at_lf_'+str(lockdown_fraction)] = predicted_df['Predicted_at_lf_'+str(lockdown_fraction)].diff()
        
        predicted_df['Infected_per_day_at_lf_'+str(lockdown_fraction)][predicted_df['Infected_per_day_at_lf_'+str(lockdown_fraction)].isna()] = predicted_df['Predicted_at_lf_'+str(lockdown_fraction)][predicted_df['Infected_per_day_at_lf_'+str(lockdown_fraction)].isna()]-min_case
    
    
    predicted_df['Actual_infected_per_day'] = predicted_df['Actual'].diff()
    predicted_df['Actual_infected_per_day'][predicted_df['Actual_infected_per_day'].isna()] = predicted_df['Actual'][predicted_df['Actual_infected_per_day'].isna()]-min_case
    
    plotgraph(predicted_df,cityname=cityname, days=days, min_case = min_case)
    
    plotgraph2(predicted_df,cityname=cityname, days=days, min_case = min_case)
    
    predicted_cols = [s for s in predicted_df.columns if 'Predicted' in s]
    
    error_cols = [s for s in predicted_df.columns if 'Error' in s]
    
    best_beta_cols = [s for s in predicted_df.columns if 'best_beta' in s]
    
    qf_cols = [s for s in predicted_df.columns if 'best_qf' in s] 
    
    Ro_num_cols = [s for s in predicted_df.columns if 'Ro_num' in s]
    
    infected_cols = [s for s in predicted_df.columns if 'Infected_at_lf' in s]
    
    recovered_cols = [s for s in predicted_df.columns if 'Recovered' in s]
    
    infectedperday_cols = [s for s in predicted_df.columns if 'Infected_per_day' in s]
    
    predicted_df = predicted_df[['City', 'CityPopulation', 'Date', 'Days','incubation_period', 'infectious_period','Actual'] + predicted_cols + infectedperday_cols +qf_cols+best_beta_cols + Ro_num_cols+error_cols + recovered_cols + infected_cols + ['Actual_infected_per_day']] 
    predicted_df.rename(columns={'City':'Region','CityPopulation':'RegionPopulation'},inplace=True)
    
    predicted_df.to_excel(cityname+'_results.xlsx',index=False)
    
    os.chdir(working_directory)
    
###############################################################################
    
# Main SEIR model prediction
    
Ro_restricted_list = [2.3,4]

for Ro_restricted in Ro_restricted_list:

    # Iterate over different prediction region levels - Country, State, City
    for prediction_level in select_prediction_level:
        
        # Get list of regions
        reqd_city_df = reqd_city_df_orig[reqd_city_df_orig['City/State/Country']==prediction_level]
    
        # Iterating Through all reqd regions to predict the number of cases in coming weeks
        for index, row in reqd_city_df.iterrows():
            
            #Changing Directory to output folder
            current_directory = os.getcwd()
            
            new_directory = os.path.join(current_directory,"Ro_restricted_to_"+str(Ro_restricted),prediction_level+" level results")
            if not os.path.exists(new_directory):
               os.makedirs(new_directory)
            os.chdir(new_directory)    
        
            # City Name
            cityname = row['Place']
            #City Population
            citypop = int(row['Population'])
            #Initial number of cases
            min_case = int(row['min_case'])
            #Edit Period of Incubation
            incubation_period=5.1
            #Edit Infectious Period
            infectious_period=2.3
            #Flag for State. Keep it as True for Delhi and Kerala, For Others keep it as False
            State_Flag = row['City/State/Country'] == 'State'
            # No of days to predict from future
            no_of_future_days = 180
            #EbyIratio : Exposed to Infected Ratio    
            EbyIratio = 4
            
            
            #Filering Data for Reqd city.
            if(prediction_level=="Country"):
                city_raw_df = raw_df.copy(deep=True)
            else:
                if(State_Flag):
                    city_raw_df = raw_df[(raw_df['detectedstate']== cityname)]
                else:
                    city_raw_df = raw_df[(raw_df['detecteddistrict']== cityname)]    
                
            #Changing Date Format
            city_raw_df['dateannounced'] = pd.to_datetime(city_raw_df['dateannounced'], format='%d/%m/%Y')
               
            city_raw_df['dateannounced'] = city_raw_df['dateannounced'].apply(lambda x: x.date())
               
            city_cases_df = get_cummulative_count(city_raw_df)
                
            city_cases_df = impute_missing_dates(city_cases_df)
            
            city_df = basicplots(city_cases_df,min_case)
                
            city_df.reset_index(drop=True,inplace=True)
                
            no_of_days = len(city_df)
                
            cityinfected = city_df.Cases.iloc[0]
                
            # lockdown Fraction 
            lockdown_fraction_list = [0.0,0.3,0.5]
            # quarantine Min Fraction
            quarantinemin = 0 
            # quarantine Max Fraction
            quarantinemax = 0.6 
            
            #Beta Min
            betamin = round(1 / incubation_period,2) #min beta value to get Ro value greater than 1
            # Beta Max
            betamax = round(Ro_restricted / incubation_period,2) #1.0 -> max beta value to get Ro within Ro_restricted
                
            # To get the best pairs of lockdownfraction & beta values with minimum error
            best_params_df = grid_search(city_df,citypop, cityinfected,days= (no_of_days-1),lockdown_fraction_list = lockdown_fraction_list,quarantinemin=quarantinemin,quarantinemax=quarantinemax, 
                                         betamax=betamax,betamin=betamin,incubation_period=incubation_period, infectious_period=infectious_period,
                                         test_days=2,overpredict_percent = 2,EbyIratio=EbyIratio)
                
            # Write Results
            write_results(city_df,best_params_df,cityname,citypop,cityinfected,min_case=min_case,days=(no_of_days+no_of_future_days-1),EbyIratio=EbyIratio)
    
###############################################################################
