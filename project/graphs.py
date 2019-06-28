import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'tra': pd.read_csv('air_visit_data.csv'),
    'as': pd.read_csv('air_store_info.csv'),
    'hs': pd.read_csv('hpg_store_info.csv'),
    'ar': pd.read_csv('air_reserve.csv'),
    'hr': pd.read_csv('hpg_reserve.csv'),
    'id': pd.read_csv('store_id_relation.csv'),
    'tes': pd.read_csv('sample_submission.csv'),
    'hol': pd.read_csv('date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }
plt.style.use('ggplot')

colo = ['orange','lightsalmon','tomato','indianred','darkslategrey','darkslateblue','indigo','darkmagenta','crimson']



data['ar']['visit_date'] = data['ar']['visit_datetime'].map(lambda x: str(x).split(' ')[0])
data["ar"] = data["ar"].drop("visit_datetime",axis=1)
data["ar"] = data["ar"].drop("reserve_datetime",axis=1)

ar1 = data['ar'].groupby(["air_store_id",'visit_date'],as_index=False)['reserve_visitors'].sum()
# ar1.to_csv("und/121.csv", index=False)
ar1 = pd.merge(ar1, data['id'], how='left', on=['air_store_id'])
# ar1.to_csv("und/121.csv", index=False)

tra1 = pd.merge(data['tra'], ar1, how='left', on=['air_store_id','visit_date'])

data['hr']['visit_date'] = data['hr']['visit_datetime'].map(lambda x: str(x).split(' ')[0])
data["hr"] = data["hr"].drop("visit_datetime",axis=1)
data["hr"] = data["hr"].drop("reserve_datetime",axis=1)
hr1 = data['hr'].groupby(["hpg_store_id",'visit_date'],as_index=False)['reserve_visitors'].sum()
tra2 = pd.merge(tra1, hr1, how='left', on=['hpg_store_id','visit_date'])

tra2['reserve_visitors_x'] = tra2['reserve_visitors_x'].fillna(0)
tra2['reserve_visitors_y'] = tra2['reserve_visitors_y'].fillna(0)
tra2['reserve'] = tra2['reserve_visitors_x'] + tra2['reserve_visitors_y']
tra2 = tra2.drop("reserve_visitors_x",axis=1)
tra2 = tra2.drop("reserve_visitors_y",axis=1)
tra2 = tra2.drop("hpg_store_id",axis=1)
# tra2.to_csv("122.csv", index  =False)
z0 = tra2.groupby(['visit_date'], as_index=False)['reserve'].sum()
plt.plot(z0['visit_date'],z0['reserve'])
plt.xticks([])
plt.xlabel("Visit Date")
plt.ylabel("Reservations")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()

def estimate_coef(x, y): 
    n = np.size(x) 
   
    m_x, m_y = np.mean(x), np.mean(y) 
   
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
   
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 

    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
    plt.xlim(0,120)
    plt.ylim(0,300) 
    y_pred = b[0] + b[1]*x
    er = np.sum((y-y_pred)*(y-y_pred)) 
    er = er/(x.shape[0])
    er = er**0.5
    plt.plot(x, y_pred, color = "g") 
    plt.xlabel('No. Of Reservations') 
    plt.ylabel('Visitors') 
    figManager = plt.get_current_fig_manager()
    figManager.window.state("zoomed")
    plt.show() 
    return er
  
b = estimate_coef(tra2['reserve'],tra2['visitors']) 
er =  plot_regression_line(tra2['reserve'],tra2['visitors'], b) 
print(b[0],b[1],er)

for i in range(0,tra2['reserve'].shape[0]):
    if tra2['reserve'][i]!=0:
        ert = b[0] + b[1]*tra2['reserve'][i]
        if ert> (tra2['visitors'][i] + er ):
            tra2['reserve'][i] = 0

er =  plot_regression_line(tra2['reserve'],tra2['visitors'], b) 



data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])

z1 = data['tra'].groupby(['visit_date'], as_index=False)['visitors'].sum()
plt.plot(z1['visit_date'],z1['visitors'])
plt.xlabel("Visit Date")
plt.ylabel("Visitors")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()
# z1.to_csv("und/1.csv", index=False)

z2 = train.groupby(['day_of_week'],as_index=False)['visitors'].mean()
# z2.to_csv("und/3.csv", index=False)
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
z2 = z2.set_index('day_of_week').loc[order]

plt.bar(order,z2['visitors'],color=colo)
plt.xlabel("Day")
plt.ylabel("Avg-visitors")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()

z10 = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
z10 = z10.groupby(['visit_date','day_of_week'],as_index=False)['visitors'].mean()
# z10.to_csv("213.csv",index=False)

order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
z11 = z10.set_index('day_of_week').loc[order]
z11 = pd.merge(z10, z11, how='left', on=['visit_date','visitors']) 
sns.boxplot(z11["day_of_week"],z11['visitors'],order=order)

plt.xlabel("Day")
plt.ylabel("Visitors")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()



z3 = train.groupby(['month'],as_index=False)['visitors'].mean()
plt.bar(z3['month'],z3['visitors'],color=colo)
plt.xlabel("Month")
plt.ylabel("Avg-visitors")
# plt.savefig("month-meanvisitors.png")
# name_month = ["Jan","Feb","March","Apr","May","June","July","Sep","Oct","Nov","Dec"]
# plt.xticks([name_month)
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()
# z3.to_csv("und/4.csv", index=False)

z4 = data['as'].groupby(['air_genre_name'], as_index=False)['air_store_id'].count()
plt.barh(z4['air_genre_name'],z4['air_store_id'],color=colo)
# plt.savefig("month-meanvisitors.png")
plt.xlabel("No. of stores")
plt.ylabel("Genre")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()
# z4.to_csv("und/5.csv",index=False)

z5 = data['as'].groupby(['air_area_name'], as_index=False)['air_store_id'].count()
plt.barh(z5['air_area_name'],z5['air_store_id'],color=colo)
# plt.savefig("month-meanvisitors.png")
plt.xlabel("No. of stores")
plt.ylabel("Area")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()

# z5.to_csv("und/6.csv",index=False)

z6 = data['hol'].groupby(['holiday_flg'], as_index=False)['day_of_week'].count()
# z6.to_csv("und/7.csv",index=False)
plt.bar(z6['holiday_flg'],z6['day_of_week'],color=colo)
plt.xlabel("Holiday_flg")
plt.ylabel("Count")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
# plt.savefig("month-meanvisitors.png")
plt.show()


z7= pd.merge(data['tra'],data['as'],how="left",on="air_store_id")
# z7.to_csv("und/118.csv",index=False)

z7 = z7.groupby(["air_genre_name","visit_date"],as_index=False)["visitors"].mean()
# z7.to_csv("und/119.csv",index=False)

z8 = train.groupby(['day_of_week','holiday_flg'],as_index=False)['visitors'].mean()
# z8.to_csv("und/8.csv",index=False)

order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
z9 = z8.set_index('day_of_week').loc[order]
z9 = pd.merge(z9,z8,how="left",on=["visitors","holiday_flg"])
z9.to_csv("he.csv",index=False)
plt.scatter(z9['day_of_week'],z9['visitors'])
plt.xlabel("Day of Week")
plt.ylabel("Avg Visitors")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()

unique = z7["air_genre_name"].unique()
count =0
prev = 0
fig, axs = plt.subplots(ncols = 4, nrows = 4,sharex=True,sharey= True)
for i in unique:
    z = []
    z = z7["air_genre_name"][z7["air_genre_name"]==i].index
    ze = z.shape[0] + prev
    axs[int(count/4)][int(count%4)].plot(z7["visit_date"][prev:ze],(z7["visitors"][prev:ze]), color = colo[count%9])
    axs[int(count/4)][int(count%4)].set(title=i)
    count = count +1
    prev = ze
plt.xticks([])
fig.suptitle("No. of Visitors v/s Date")
plt.yscale('log')
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()


