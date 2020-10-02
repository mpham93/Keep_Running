import pandas as pd
from sklearn import preprocessing

def filter_id_date (df, IDs, SubscriptionHistory, duration=7, add=True):
    df_id = pd.DataFrame(columns = df.columns)
    for ID in IDs:
        trial_start = SubscriptionHistory[(SubscriptionHistory['user']==ID)&(SubscriptionHistory['event']=='subscription-free-trial')]['eventDate']
        if add == True:
            trial_end = trial_start + pd.DateOffset(days=duration)
            trial_start = trial_start.values[0]
            trial_end = trial_end.values[0]
            selected_row = df[(df['user_id'] == ID)&(df['received_at']>=trial_start)&(df['received_at']<trial_end)]
        elif add == False:
            begin = trial_start - pd.DateOffset(days=duration)
            trial_start = trial_start.values[0]
            begin = begin.values[0]
            selected_row = df[(df['user_id'] == ID)&(df['received_at']<trial_start)&(df['received_at']>=begin)]
        df_id = df_id.append(selected_row)
    # print (f'original shape {df.shape}, after filtering {df_id.shape}')
    return df_id

def max_consecutive_days (df):
    user_day = {}
    IDs = df.user_id.unique()
    for i in IDs:
        df_id = df[df.user_id == i]
        df_id = df_id.sort_values(by=['received_at'])
        dt = df_id.received_at
        day = pd.Timedelta('1d')
        in_block = ((dt - dt.shift(-1)).abs() == day) | (dt.diff() == day)
        filt = df_id.loc[in_block]
        breaks = filt['received_at'].diff() != day
        groups = breaks.cumsum()
        cnt = filt.groupby(groups).agg('count').received_at.max()
        user_day[i] = cnt
    df_cnt = pd.DataFrame(list(user_day.items()), columns = ['user_id','number_consecutive_day']).set_index(['user_id'])
    df_cnt = df_cnt.fillna(1)
    return df_cnt

def feature_duration(DFs, IDs, SubscriptionHistory, duration = 7, add = True, filter = True, consecutive = True, freq = True, cal_ave = False, dayDF = pd.DataFrame()):
    ActivityComplete, SessionCompleted, LibrarySelectedTab, ScreenViewed, AppForeground, SessionViewed = DFs
    results = {'ActivityComplete':{}, 'SessionCompleted': {}, 'LibrarySelectedTab': {}, 'ScreenViewed': {}, 'AppForeground':{}, 'SessionViewed' :{}}
    ### filter events by dates ###
    if filter == True:
        ActivityComplete_filter   = filter_id_date(ActivityComplete, IDs, SubscriptionHistory, duration=duration, add = add)
        SessionCompleted_filter   = filter_id_date(SessionCompleted, IDs, SubscriptionHistory, duration=duration, add = add)
        LibrarySelectedTab_filter = filter_id_date(LibrarySelectedTab, IDs, SubscriptionHistory, duration=duration, add = add)
        ScreenViewed_filter       = filter_id_date(ScreenViewed, IDs, SubscriptionHistory, duration=duration, add = add)
        AppForeground_filter      = filter_id_date(AppForeground, IDs, SubscriptionHistory, duration=duration, add = add)
        SessionViewed_filter      = filter_id_date(SessionViewed, IDs, SubscriptionHistory, duration=duration, add = add)
    else:
        ActivityComplete_filter   = ActivityComplete
        SessionCompleted_filter   = SessionCompleted
        LibrarySelectedTab_filter = LibrarySelectedTab
        ScreenViewed_filter       = ScreenViewed
        AppForeground_filter      = AppForeground
        SessionViewed_filter      = SessionViewed

    results['ActivityComplete']['filter'] = ActivityComplete_filter
    results['SessionCompleted']['filter'] = SessionCompleted_filter
    results['LibrarySelectedTab']['filter'] = LibrarySelectedTab_filter
    results['ScreenViewed']['filter'] = ScreenViewed_filter
    results['AppForeground']['filter'] = AppForeground_filter
    results['SessionViewed']['filter'] = SessionViewed_filter

    ### find the maximum number of consecutive days with continuous app engagement ###
    if consecutive == True:
        results['ActivityComplete']['cons'] = max_consecutive_days(ActivityComplete_filter)
        results['SessionCompleted']['cons'] = max_consecutive_days(SessionCompleted_filter)
        results['LibrarySelectedTab']['cons'] = max_consecutive_days(LibrarySelectedTab_filter)
        results['ScreenViewed']['cons'] = max_consecutive_days(ScreenViewed_filter)
        results['AppForeground']['cons'] = max_consecutive_days(AppForeground_filter)
        results['SessionViewed']['cons'] = max_consecutive_days(SessionViewed_filter)

    ### count the frequency/number of times of app engagement ###
    if freq == True:
        results['ActivityComplete']['freq']   = ActivityComplete_filter.groupby(['user_id']).agg(['count'])['received_at']
        results['SessionCompleted']['freq']   = SessionCompleted_filter.groupby(['user_id']).agg(['count'])['received_at']
        results['LibrarySelectedTab']['freq']   = LibrarySelectedTab_filter.groupby(['user_id']).agg(['count'])['received_at']
        results['ScreenViewed']['freq']   = ScreenViewed_filter.groupby(['user_id']).agg(['count'])['received_at']
        results['AppForeground']['freq']      = AppForeground_filter.groupby(['user_id']).agg(['count'])['received_at']
        results['SessionViewed']['freq']      = SessionViewed_filter.groupby(['user_id']).agg(['count'])['received_at']
    
    ### calculate average #activities/day ###
    if cal_ave == True:
        def ave_Act(df, dayDF=dayDF):
            m = df.join(dayDF)
            m.loc[m[m['days_create2Trial']>duration].index,'days_create2Trial'] = duration 
            m['daily_ave'] = m['count']/m['days_create2Trial']
            return m
        def normalize_df (df, column='daily_ave'):
            x = df[column].values.reshape(-1, 1)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df[('normed_'+column)] = x_scaled
            return df
        
        results['ActivityComplete']['ave'] = ave_Act(results['ActivityComplete']['freq'])
        results['SessionCompleted']['ave'] = ave_Act(results['SessionCompleted']['freq'])
        results['LibrarySelectedTab']['ave'] = ave_Act(results['LibrarySelectedTab']['freq'])
        results['ScreenViewed']['ave'] = ave_Act(results['ScreenViewed']['freq'])
        results['AppForeground']['ave'] = ave_Act(results['AppForeground']['freq'])
        results['SessionViewed']['ave'] = ave_Act(results['SessionViewed']['freq'])

        results['ActivityComplete']['norm'] = normalize_df(results['ActivityComplete']['ave'])
        results['SessionCompleted']['norm'] = normalize_df(results['SessionCompleted']['ave'])
        results['LibrarySelectedTab']['norm'] = normalize_df(results['LibrarySelectedTab']['ave'])
        results['ScreenViewed']['norm'] = normalize_df(results['ScreenViewed']['ave'])
        results['AppForeground']['norm'] = normalize_df(results['AppForeground']['ave'])
        results['SessionViewed']['norm'] = normalize_df(results['SessionViewed']['ave'])

    return results


