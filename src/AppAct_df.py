import pandas as pd 

def process_AppAct_df (data_dir = '../data/S3/AppEventData/'):
    ### read csv files ###
    CommunityMember = pd.read_csv(data_dir+'CommunityMember.csv', sep=',', quotechar='"', index_col=0)
    SubscriptionHistory = pd.read_csv(data_dir+'SubscriptionHistory.csv', sep=',', quotechar='"', index_col=0)

    iosActivityComplete = pd.read_csv(data_dir+'ios.activity_complete.csv', sep=',', quotechar='"', index_col=0)
    iosSessionCompleted = pd.read_csv(data_dir+'ios.session_completed.csv', sep=',', quotechar='"', index_col=0)
    iosLibrarySelectedTab = pd.read_csv(data_dir+'ios.library_selected_tab.csv', sep=',', quotechar='"', index_col=0)
    iosScreenViewed = pd.read_csv(data_dir+'ios.screen_viewed.csv', sep=',', quotechar='"', index_col=0)
    iosAppForeground = pd.read_csv(data_dir+'ios.app_enter_foreground.csv', sep=',', quotechar='"', index_col=0)
    iosSessionViewed = pd.read_csv(data_dir+'ios.session_element_viewed.csv', sep=',', quotechar='"', index_col=0)
    
    androidActivityComplete   = pd.read_csv(data_dir+'ongo_android.activity_complete.csv', sep=',', quotechar='"', index_col=0)
    androidSessionCompleted   = pd.read_csv(data_dir+'ongo_android.session_completed.csv', sep=',', quotechar='"', index_col=0)
    androidLibrarySelectedTab = pd.read_csv(data_dir+'ongo_android.library_selected_tab.csv', sep=',', quotechar='"', index_col=0)
    androidScreenViewed       = pd.read_csv(data_dir+'ongo_android.screen_viewed.csv', sep=',', quotechar='"', index_col=0)
    androidAppForeground      = pd.read_csv(data_dir+'ongo_android.app_enter_foreground.csv', sep=',', quotechar='"', index_col=0)
    androidSessionViewed = pd.read_csv(data_dir+'ongo_android.session_element_viewed.csv', sep=',', quotechar='"', index_col=0)

    treActivityComplete   = pd.read_csv(data_dir+'tre.activity_complete.csv', sep=',', quotechar='"', index_col=0)
    treSessionCompleted   = pd.read_csv(data_dir+'tre.session_completed.csv', sep=',', quotechar='"', index_col=0)
    treScreenViewed       = pd.read_csv(data_dir+'tre.screen_viewed.csv', sep=',', quotechar='"', index_col=0)
    treSessionViewed = pd.read_csv(data_dir+'tre.session_element_viewed.csv', sep=',', quotechar='"', index_col=0)
    
    ### format date data type ###
    CommunityMember.createdAt = pd.to_datetime(CommunityMember.createdAt).dt.normalize()
    SubscriptionHistory.eventDate = pd.to_datetime(SubscriptionHistory.eventDate)
    SubscriptionHistory.eventDate = SubscriptionHistory.eventDate.dt.normalize()
    iosActivityComplete['received_at'] = pd.to_datetime(iosActivityComplete['received_at'])
    iosSessionCompleted['received_at'] = pd.to_datetime(iosSessionCompleted['received_at'])
    iosLibrarySelectedTab['received_at'] = pd.to_datetime(iosLibrarySelectedTab['received_at'])
    iosScreenViewed['received_at'] = pd.to_datetime(iosScreenViewed['received_at'])
    iosAppForeground['received_at'] = pd.to_datetime(iosAppForeground['received_at'])
    iosSessionViewed['received_at'] = pd.to_datetime(iosSessionViewed['received_at'])

    androidActivityComplete['received_at']   = pd.to_datetime(androidActivityComplete['received_at']).dt.normalize()
    androidSessionCompleted['received_at']   = pd.to_datetime(androidSessionCompleted['received_at']).dt.normalize()
    androidLibrarySelectedTab['received_at'] = pd.to_datetime(androidLibrarySelectedTab['received_at']).dt.normalize()
    androidScreenViewed['received_at']       = pd.to_datetime(androidScreenViewed['received_at']).dt.normalize()
    androidAppForeground['received_at']      = pd.to_datetime(androidAppForeground['received_at']).dt.normalize()
    androidSessionViewed['received_at'] = pd.to_datetime(androidSessionViewed['received_at'])

    treActivityComplete['received_at']   = pd.to_datetime(treActivityComplete['received_at']).dt.normalize()
    treSessionCompleted['received_at']   = pd.to_datetime(treSessionCompleted['received_at']).dt.normalize()
    treScreenViewed['received_at']       = pd.to_datetime(treScreenViewed['received_at']).dt.normalize()
    treSessionViewed['received_at'] = pd.to_datetime(treSessionViewed['received_at'])

    ### merge data tables from ios, android, and tre ###
    ActivityComplete = pd.concat([iosActivityComplete, androidActivityComplete, treActivityComplete])
    SessionCompleted = pd.concat([iosSessionCompleted, androidSessionCompleted, treSessionCompleted])
    LibrarySelectedTab = pd.concat([iosLibrarySelectedTab, androidLibrarySelectedTab])
    ScreenViewed = pd.concat([iosScreenViewed, androidScreenViewed, treScreenViewed])
    AppForeground = pd.concat([iosAppForeground, androidAppForeground])
    SessionViewed = pd.concat([iosSessionViewed, androidSessionViewed, treSessionViewed])
    
    return CommunityMember, SubscriptionHistory, ActivityComplete, SessionCompleted, LibrarySelectedTab, ScreenViewed, AppForeground, SessionViewed