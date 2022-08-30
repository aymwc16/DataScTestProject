"""
This python file regroups classes with built-in methods
to simplify the plotting process in the final data analysis
"""

### Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re
import sys
import os

class DataFrameBuild:
    def __init__(self, csv_file_path) -> None:
        self.csv_path = csv_file_path
        self.csv_name = re.findall("data\/(.*)csv$",csv_file_path)[0]
        self.day_number = int(re.findall("\d+",csv_file_path)[0])
    

    def rotation_matrix_in_frame(self,Q:np.array) -> np.array:
        """Computes the rotation matrix in correct frame
        Q: rotation matrix"""

        q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]
        R = np.array([[1-2*(q2**2+q3**2)    , 2*(q1*q2-q0*q3)   , 2*(q0*q2+q1*q3)],
                      [2*(q1*q2+q0*q3)      , 1-2*(q1**2+q3**2) , 2*(q2*q3-q0*q1)],
                      [2*(q1*q3-q0*q2)      , 2*(q0*q1+q2*q3)   , 1-2*(q1**2+q2**2)]])
        return R
    

    def frame_projection(self,df:pd.DataFrame,verbose=False) -> pd.DataFrame:
        """Projects the speed in inertial frame into the correct frame

        df: dataframe we are working with
        verbose: print detailed console instruction"""

        speed_in_frame = []
        print("Frame projection with linewise computation for {}csv...".format(self.csv_name))
        for j in range(df.shape[0]):
            if j%2500==0 and j!=0 and verbose: # Status printed in console
                print("[{}]\tlinewise computation for {}csv... \tProgress: {:.1f}%".format(dt.datetime.utcnow(),
                                                                                           self.csv_name,
                                                                                           j/df.shape[0]*100))
            R = self.rotation_matrix_in_frame(df.loc[j,'quaternion1':'quaternion4'].to_numpy())
            speed_in_frame.append(R@df.loc[j,'speed1':'speed3'].to_numpy().T)
            df_speeds = pd.DataFrame(speed_in_frame).rename(columns={0: "frame_speed1",1: "frame_speed2",2: "frame_speed3"})
        print("... projection finished for {}csv!".format(self.csv_name))
        return df_speeds


    def specific_kinetic_energy(self, df:pd.DataFrame, parameter='total') -> pd.DataFrame:
        """Computes the specific kinetic energy of the object using speeds
        registered in the dataset.

        df: dataframe we are working with
        parameter: must have one of those values ('all','total','1','2',3')
            \t'1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            \t'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            \t'all':\tcomputes '1', '2', '3', and 'total'
        """

        print("[{}]\ Computation of kinetic energy {} for {}csv".format(dt.datetime.utcnow(),
                                                                        parameter,
                                                                        self.csv_name))
        if parameter == 'total':
            return 0.5*(df['frame_speed1']**2+df['frame_speed2']**2+df['frame_speed3']**2)
        if parameter in ['1','2','3']:
            return 0.5*(df['frame_speed{}'.format(parameter)]**2)
        if parameter == 'all':
            list_param = ['total','1','2','3']
            pd.concat([self.specific_kinetic_energy(df, parameter=param) for param in list_param],axis=1)
        else:
            raise ValueError("Error: Wrong paramater value! 'parameter' must be in ['total','1','2','3']")
    

    def specific_power_computation(self, df:pd.DataFrame, parameter='total') -> pd.DataFrame:
        """Computes the specific power of the object using speeds
        registered in the dataset.

        df: dataframe we are working with
        parameter: must have one of those values ('all','total','1','2',3')
            \t'1', '2', '3':\tcomputes the specific power associated to speeds(1/2/3)
            \t'total':\tcomputes the sum of total specific powers associated to total speeds
            \t'all':\tcomputes '1', '2', '3', and 'total'
        """

        print("[{}]\tComputation of power {} for {}csv".format(dt.datetime.utcnow(),
                                                               parameter,
                                                               self.csv_name))
        if parameter == 'all':
            return self.specific_kinetic_energy(df,parameter=parameter).diff()/((df.iloc[-1,0]-df.iloc[0,0])/df.shape[0])
        if parameter in ['total','1','2','3']:
            return self.specific_kinetic_energy(df,parameter=parameter).diff()/((df.iloc[-1,0]-df.iloc[0,0])/df.shape[0])


    def build_dataframe(self,save=True,force=False,verbose=False) -> pd.DataFrame:
        """Automated dataframe generation that can be stored as a csv.

        save: save the dataset into a .csv
        force: overwrite the dataset if it was previously generated and builds another one
        verbose: print more messages in console"""

        """if 'processed' not in self.csv_path:
            processed_csv_path = './../data/'+self.csv_name[:-1].lower()+'_processed.csv'
        else:
            processed_csv_path = self.csv_path"""

        if 'processed' in self.csv_path: # avoids building an existing dataset
            if os.path.isfile(self.csv_path) and not force:
                print("{} was already built. Retrieving...".format(self.csv_path))
                return pd.read_csv(self.csv_path)
            else:
                sys.exit()
        else:
            processed_csv_path = './../data/'+self.csv_name[:-1].lower()+'_processed.csv'
            if os.path.isfile(processed_csv_path):
                pass
            else:
                # Reading the csv dataset
                df = pd.read_csv(self.csv_path)
                print("{}csv dataframe retrieved".format(self.csv_name)) # status in console

                #Filtering any speed that is above 6m/s
                for i in range(1,4):
                    df['speed{}'.format(i)].mask(np.abs(df['speed{}'.format(i)])>6,0,inplace=True)

                # Computing the time windows (hours, day, week) out of seconds
                df['time'] = df['time'].round()
                df['hour'] = df['time'].apply(lambda x: x//3600+8)
                df['day'] = df['time'].apply(lambda x: self.day_number)
                df['week'] = df['time'].apply(lambda x: (self.day_number-1)//7+1)

                # Adjusting granularity and formatting data for readibility
                df_compressed = df.groupby('time').agg(np.mean).reset_index()
                df_compressed['formatted_time'] = df_compressed['time'].apply(lambda x: str(dt.timedelta(seconds=(x+8*3600))))
                df_compressed.drop(columns=['Unnamed: 0'], inplace=True)

                df_speeds_in_frame = self.frame_projection(df_compressed,verbose)
                df_joined = df_compressed.join(df_speeds_in_frame,how='inner')

                projections = ['total','1','2','3']
                for param in projections:
                    df_joined['specific_power_{}'.format(param)] = self.specific_power_computation(df_joined,parameter=param)
                print("... dataframe built for {}csv!".format(self.csv_name))
                print("Dataframe info...\nEntries:\t{}\nShape:\t {}".format(df_joined.columns,df_joined.shape))

                # Saving processed dataframe into a csv
                if save:
                    df_joined.to_csv(processed_csv_path)
                    print("[{}]\tCurated dataframe of {}csv stored in {}".format(dt.datetime.utcnow(),
                                                                                    self.csv_name,
                                                                                    processed_csv_path))
                return df_joined

class PlotTool:
    """Offers built-in methods to simplify the plotting process"""

    def __init__(self,df_list:list) -> None:
        self.df_ref = df_list
        self.df_study = df_list
        self.palette = sns.color_palette("rocket", 15)
        self.legends = []
        plt.figure(figsize=(12,7),dpi=100)


    def plot_hourly_power(self,unique=False,day_number=1,parameter='total'):
        """Generate variations of clinical variations by the hour for plotting

        unique: specify is there one day to plot or not
        day_number: indicates the day to plot in range [1,15] if 'unique' is specified
        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        if unique:
            self.df_study = self.df_ref[day_number-1]
            sns.scatterplot(data = self.df_study, 
                            x = 'hour',
                            y = 'specific_power_{}'.format(parameter), 
                            palette = self.palette,
                            alpha = 0.5)
            self.legends.append("Day "+str(day_number))
            plt.title("Hourly Variations of specific_power_{}".format(parameter))
        else:
            self.df_study = pd.concat(self.df_ref,axis=0).reset_index()
            sns.scatterplot(data = self.df_study, x = 'hour', y = 'specific_power_{}'.format(parameter), hue='day', palette=self.palette)
        plt.xlabel("Hour", fontsize=13)
        return None


    def plot_aggregated_hourly_power(self,unique=False,day_number=1,parameter='total'):
        """Generate hourly evolutions of clinical for plotting

        unique: specify is there one day to plot or not
        day_number: indicates the day to plot in range [1,15] if 'unique' is specified
        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        if unique:
            self.df_study = self.df_ref[day_number-1].loc[:,['hour','day','specific_power_{}'.format(parameter)]] \
                                                     .groupby(['hour','day']).agg(sum).reset_index()
            sns.lineplot(data = self.df_study, x = 'hour', y = 'specific_power_{}'.format(parameter), palette=self.palette)
        else:
            self.df_study = pd.concat(self.df_ref,axis=0).loc[:,['hour','day','specific_power_{}'.format(parameter)]] \
                                                         .groupby(['hour','day']).agg(sum).reset_index()
            sns.lineplot(data = self.df_study, x = 'hour', y = 'specific_power_{}'.format(parameter), hue='day', palette=self.palette)
        plt.title("Hourly Evolution of specific_power_{}".format(parameter))
        plt.xlabel("Hour", fontsize=13)
        return None


    def plot_daily_power(self,unique=False,day_number=1,parameter='total'):
        """Generate daily variations of clinical for plotting

        unique: specify is there one day to plot or not
        day_number: indicates the day to plot in range [1,15] if 'unique' is specified
        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        if unique:
            self.df_study = self.df_ref[day_number-1].loc[:,['hour','day','specific_power_{}'.format(parameter)]] \
                                                     .groupby(['hour','day']).agg(sum).reset_index()
            sns.scatterplot(data = self.df_study, 
                            x = 'day',
                            y = 'specific_power_{}'.format(parameter),
                            alpha = 0.5)
        else:
            self.df_study = pd.concat(self.df_ref,axis=0).loc[:,['hour','day','specific_power_{}'.format(parameter)]] \
                                                         .groupby(['hour','day']).agg(sum).reset_index()
            sns.boxplot(data = self.df_study, x = 'day', y = 'specific_power_{}'.format(parameter))
        plt.title("Daily Variations of specific_power_{}".format(parameter))
        plt.xlabel("Day", fontsize=13)
        return None
    

    def plot_aggregated_daily_power(self,unique=False,day_number=1,parameter='total'):
        """Generate weekly evolutions of clinical for plotting

        unique: specify is there one day to plot or not
        day_number: indicates the day to plot in range [1,15] if 'unique' is specified
        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        if unique:
            self.df_study = self.df_ref[day_number-1].loc[:,['day','specific_power_{}'.format(parameter)]] \
                                                     .groupby(['day']).agg(sum).reset_index()
        else:
            self.df_study = pd.concat(self.df_ref,axis=0).loc[:,['day','specific_power_{}'.format(parameter)]] \
                                                         .groupby(['day']).agg(sum).reset_index()
        sns.lineplot(data = self.df_study, x = 'day', y = 'specific_power_{}'.format(parameter), palette=self.palette)
        plt.title("Daily Variations of specific_power_{}".format(parameter))
        plt.xlabel("Day", fontsize=13)
        return None
    

    def plot_weekly_power(self,parameter='total'):
        """Generate weekly variations of clinical for plotting

        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        self.df_study = pd.concat(self.df_ref,axis=0).loc[:,['day','week','specific_power_{}'.format(parameter)]] \
                                                     .groupby(['week','day']).agg(sum).reset_index()
        sns.boxplot(data = self.df_study[self.df_study['week']<3.0], x = 'week', y = 'specific_power_{}'.format(parameter))
        plt.title("Weekly Variations of specific_power_{}".format(parameter))
        plt.xlabel("Week", fontsize=13)
        return None
    

    def plot_aggregated_weekly_power(self,parameter='total'):
        """Generate weekly evolutions of clinical for plotting

        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        self.df_study = pd.concat(self.df_ref,axis=0).loc[:,['week','specific_power_{}'.format(parameter)]] \
                                                        .groupby(['week']).agg(sum).reset_index()
        sns.lineplot(data = self.df_study[self.df_study['week']<3.0], x = 'week', y = 'specific_power_{}'.format(parameter), palette=self.palette)
        plt.title("Weekly Evolution of specific_power_{}".format(parameter))
        plt.xlabel("Week", fontsize=13)
        return None
    

    def plot(self,unique=False) -> None:
        """Generate weekly evolutions of clinical for plotting

        unique: specify if there are single days to plot or not for legend generation
        parameter: clinical parameter to plot. It must have one of those values ('all','total','1','2',3')
            '1', '2', '3':\tcomputes the specific kinetic energy associated to speed(1/2/3)
            'total':\tcomputes the sum of total specific kinetic energies associated to total speeds
            'all':\tcomputes '1', '2', '3', and 'total' 
        \n... make sure your parameter is present in the dataset!"""

        if unique:
            plt.legend(labels=self.legends)
        plt.ylabel("Specific power $\mathcal{P}_s \; (W.kg^{-1})$", fontsize=13)
        plt.show()
        return None