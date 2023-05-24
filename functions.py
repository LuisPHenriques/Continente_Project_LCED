import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Function to format y-axis labels in millions of euros
def millions_formatter(x, pos):
    return '{}M €'.format(int(x/1000000))


# Function to create a line plot for total gross and total net sales
def line_plot(dataframe, title):
    # Set the plot style and background color
    sns.set_style("ticks", {"axes.facecolor": "#fcf2f2"})

    # Create a new figure with a specific size
    plt.figure(figsize=(12, 8))
    
    # Get the first column name of the dataframe, that being the category
    columns = dataframe.columns.to_list()
    category = columns[0]

    # Create two line plots with different colors
    sns.lineplot(x=category, y="total_gross_sales", data=dataframe, zorder=10, linewidth=5, color='#84161a')
    sns.lineplot(x=category, y="total_net_sales", data=dataframe, zorder=10, linewidth=5, color='#de1c26')

    # Remove x and y-axis labels
    plt.xlabel('')
    plt.ylabel('')

    # Add a legend with custom labels
    plt.legend(labels=['Total gross sales', 'Total net sales'], fontsize=12)

    # Add a title with a custom font size
    plt.title(title, fontsize=18)

    # Remove the gridlines
    plt.grid(False)

    # Set custom font sizes for x and y-axis tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Format y-axis labels in millions of euros using a custom formatter function
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    
    # Get the y-axis limits
    ymin, ymax = plt.gca().get_ylim()
    # Set the lower limit of y-axis to 0 if it is less than 0
    if ymin < 0:
        ymin = 0
    # Otherwise, round it to the nearest 10 million
    else:
        ymin = int(ymin/10000000)

    # Round the upper limit of y-axis to the nearest 10 million
    ymax = int(ymax/10000000)
    
    # Calculate the y-axis positions for the horizontal lines at 40% and 80%
    y40 = int(ymin + (ymax-ymin)*0.3)*10000000
    y80 = int(ymin + (ymax-ymin)*0.8)*10000000
    
    # Add horizontal lines at the calculated y-axis positions
    plt.axhline(y=y40, color='black', zorder=5, linewidth=0.5)
    plt.axhline(y=y80, color='black', zorder=5, linewidth=0.5)

    # Remove the top and right spines of the plot
    sns.despine()

    # Display the plot
    plt.show()
    
    return None


def bar_plot(dataframe, title):
    # Set the plot style and background color
    sns.set_style("ticks", {"axes.facecolor": "#fcf2f2"})

    # Create a new figure with a specific size
    plt.figure(figsize=(12, 8))
    
    # Get the first column name of the dataframe, that being the category
    columns = dataframe.columns.to_list()
    category = columns[0]
    numeric = columns[1]

    # Create bars
    sns.barplot(x=category, y=numeric, data=dataframe, zorder=10, color='#de1c26')

    # Remove x and y-axis labels
    plt.xlabel('')
    plt.ylabel('')

    # Add a title with a custom font size
    plt.title(title, fontsize=18)

    # Remove the gridlines
    plt.grid(False)

    # Set custom font sizes for x and y-axis tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove the top and right spines of the plot
    sns.despine()

    # Display the plot
    plt.show()
    
    return None


def filter_customers(df):

    df_copy = df.copy()
    
    df_copy['month_number'] = 24 - ((df_copy['YEAR'] - 2021) * 12 + df_copy['MONTH'] - 1)
    
    df_copy['num_transactions'] = df_copy.groupby('CUSTOMER_ACCOUNT_NR_MASK')['TRANSACTION_ID_MASK'].transform('nunique')

    # Keep only the first row for each customer account
    df_copy.drop_duplicates(subset='CUSTOMER_ACCOUNT_NR_MASK', keep='first', inplace=True)

    # Compute the number of transactions per month and convert to integer
    df_copy['transactions_per_month'] = np.floor(df_copy['num_transactions'].div(df_copy['month_number'])).astype(int)

    # Filter customers based on transactions per month
    cust_array = df_copy[(df_copy['transactions_per_month'] > 0) & (df_copy['transactions_per_month'] < 8)][['CUSTOMER_ACCOUNT_NR_MASK']].values.flatten()

    df = df[df['CUSTOMER_ACCOUNT_NR_MASK'].isin(cust_array)]

    return df


def semester(month):
    if month in [1, 2, 3, 4, 5, 6]:
        return 1
    else:
        return 2
    

def count(customer_categories):
    count = 0
    counts = []
    for category in customer_categories:
        count += 1
        counts.append(count)
    return counts


def count_unique_transactions(customer_transactions):
    prev_transaction = None
    count = 0
    counts = []
    for transaction in customer_transactions:
        if transaction != prev_transaction:
            count += 1
        counts.append(count)
        prev_transaction = transaction
    return counts


def count_unique(customer_categories):
    prev_categories = []
    count = 0
    counts = []
    for category in customer_categories:
        if category not in prev_categories:
            count += 1
        counts.append(count)
        prev_categories.append(category)
    return counts


def calculate_rolling_avg(df_transactions, interval, feature, f_type):
    
    if f_type == 'CUST':
        if interval == 'YEAR':
            customer_groups = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR'])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR'])['TIME_KEY'].shift()
        else:
            customer_groups = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', interval])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', interval])['TIME_KEY'].shift()
        
        df_transactions['CUST_DAYS_SINCE_LAST_TRANSACTION'] = (df_transactions['TIME_KEY'] - df_transactions['LAST_TIME_KEY']).dt.days.fillna(-1)
        df_transactions['CUST_DAYS_SINCE_LAST_TRANSACTION'] = df_transactions['CUST_DAYS_SINCE_LAST_TRANSACTION'].astype(int)
        df_transactions = df_transactions.drop(columns=['LAST_TIME_KEY'])
        
        for name, group in customer_groups:
            # Calculate the rolling average for each customer
            rolling_avg = group['CUST_DAYS_SINCE_LAST_TRANSACTION'].rolling(window=1, min_periods=1).mean()
            # Reset the rolling average to 0 for the first transaction of each customer
            rolling_avg.iloc[0] = 0
            # Set the rolling average values back into the original dataframe
            df_transactions.loc[group.index, 'rolling_avg_days_since_prior_transaction'] = rolling_avg.values
        # Calculate the sum and count of the rolling average for each customer
        if interval == 'YEAR':
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR'])['rolling_avg_days_since_prior_transaction'].cumsum()
        else:
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', interval])['rolling_avg_days_since_prior_transaction'].cumsum()
        # Calculate the customer average for the rolling average
        df_transactions[feature+'_'+interval] = abs(df_transactions['rolling_avg_days_since_prior_transaction_sum'] / (df_transactions['CUST_NUM_TRANSACTIONS'+'_'+interval]-1))
        # Round the customer average to 2 decimal places
        df_transactions[feature+'_'+interval] = df_transactions[feature+'_'+interval].round(2).fillna(0)
        df_transactions = df_transactions.drop(columns=['rolling_avg_days_since_prior_transaction','rolling_avg_days_since_prior_transaction_sum'])

    elif f_type == 'SUBCAT':
        if interval == 'YEAR':
            customer_groups = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR'])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR'])['TIME_KEY'].shift()
        else:
            customer_groups = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR', interval])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR', interval])['TIME_KEY'].shift()
        
        df_transactions['SUBCAT_DAYS_SINCE_LAST_TRANSACTION'] = (df_transactions['TIME_KEY'] - df_transactions['LAST_TIME_KEY']).dt.days.fillna(-1)
        df_transactions['SUBCAT_DAYS_SINCE_LAST_TRANSACTION'] = df_transactions['SUBCAT_DAYS_SINCE_LAST_TRANSACTION'].astype(int)
        df_transactions = df_transactions.drop(columns=['LAST_TIME_KEY'])
        
        for name, group in customer_groups:
            # Calculate the rolling average for each customer
            rolling_avg = group['SUBCAT_DAYS_SINCE_LAST_TRANSACTION'].rolling(window=1, min_periods=1).mean()
            # Reset the rolling average to 0 for the first transaction of each customer
            rolling_avg.iloc[0] = 0
            # Set the rolling average values back into the original dataframe
            df_transactions.loc[group.index, 'rolling_avg_days_since_prior_transaction'] = rolling_avg.values
        # Calculate the sum and count of the rolling average for each customer
        if interval == 'YEAR':
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR'])['rolling_avg_days_since_prior_transaction'].cumsum()
        else:
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['SUBCAT_CD_EXT', 'YEAR', interval])['rolling_avg_days_since_prior_transaction'].cumsum()
        # Calculate the customer average for the rolling average
        df_transactions[feature+'_'+interval] = abs(df_transactions['rolling_avg_days_since_prior_transaction_sum'] / (df_transactions['SUBCAT_NUM_TRANSACTIONS'+'_'+interval]-1))
        # Round the customer average to 2 decimal places
        df_transactions[feature+'_'+interval] = df_transactions[feature+'_'+interval].round(2).fillna(0)
        df_transactions = df_transactions.drop(columns=['rolling_avg_days_since_prior_transaction','rolling_avg_days_since_prior_transaction_sum'])

    else:
        if interval == 'YEAR':
            customer_groups = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR'])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR'])['TIME_KEY'].shift()
        else:
            customer_groups = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', interval])
            df_transactions['LAST_TIME_KEY'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', interval])['TIME_KEY'].shift()
        
        df_transactions['CUSTSUBCAT_DAYS_SINCE_LAST_TRANSACTION'] = (df_transactions['TIME_KEY'] - df_transactions['LAST_TIME_KEY']).dt.days.fillna(-1)
        df_transactions['CUSTSUBCAT_DAYS_SINCE_LAST_TRANSACTION'] = df_transactions['CUSTSUBCAT_DAYS_SINCE_LAST_TRANSACTION'].astype(int)
        df_transactions = df_transactions.drop(columns=['LAST_TIME_KEY'])
        
        for name, group in customer_groups:
            # Calculate the rolling average for each customer
            rolling_avg = group['CUSTSUBCAT_DAYS_SINCE_LAST_TRANSACTION'].rolling(window=1, min_periods=1).mean()
            # Reset the rolling average to 0 for the first transaction of each customer
            rolling_avg.iloc[0] = 0
            # Set the rolling average values back into the original dataframe
            df_transactions.loc[group.index, 'rolling_avg_days_since_prior_transaction'] = rolling_avg.values
        # Calculate the sum and count of the rolling average for each customer
        if interval == 'YEAR':
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT','YEAR'])['rolling_avg_days_since_prior_transaction'].cumsum()
        else:
            df_transactions['rolling_avg_days_since_prior_transaction_sum'] = df_transactions.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT','YEAR', interval])['rolling_avg_days_since_prior_transaction'].cumsum()
        # Calculate the customer average for the rolling average
        df_transactions[feature+'_'+interval] = abs(df_transactions['rolling_avg_days_since_prior_transaction_sum'] / (df_transactions['CUSTSUBCAT_NUM_TRANSACTIONS'+'_'+interval]-1))
        # Round the customer average to 2 decimal places
        df_transactions[feature+'_'+interval] = df_transactions[feature+'_'+interval].round(2).fillna(0)
        df_transactions = df_transactions.drop(columns=['rolling_avg_days_since_prior_transaction','rolling_avg_days_since_prior_transaction_sum'])

    return df_transactions


def create_aggregations(df_transactions, df_ml, interval, feature, f_type, avg):
    if f_type == 'CUST':
        aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', 'MONTH', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', 'MONTH']) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', 'MONTH': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

        # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
        df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', 'MONTH'], how='left')
    
    elif f_type == 'SUBCAT':
        aggregations = df_transactions[['SUBCAT_CD_EXT', 'YEAR', 'MONTH', feature+'_'+interval]] \
                                .groupby(['SUBCAT_CD_EXT', 'YEAR', 'MONTH']) \
                                .agg({'SUBCAT_CD_EXT': 'first', 'MONTH': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

        # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
        df_ml = pd.merge(df_ml, aggregations, on=['SUBCAT_CD_EXT', 'YEAR', 'MONTH'], how='left')

    else:
        aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', 'MONTH', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', 'MONTH']) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', 'SUBCAT_CD_EXT': 'first', 'MONTH': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

        # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
        df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', 'MONTH'], how='left')

    if avg:
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].fillna(0)
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].astype(float).round(2)
    else:
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].fillna(0)
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].astype(int)    

    return df_ml


def compute_target(df_ml):
    # create a new column 'BOUGHT_SUBCAT' that indicates if a customer bought a subcategory in a given month
    df_ml['TARGET'] = (df_ml['CUSTSUBCAT_NUM_TRANSACTIONS_MONTH'] > 0).astype(int)
    
    # group the dataframe by customer_id and category_id
    grouped = df_ml.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT'])

    # create a new column 'PREV_BOUGHT_SUBCAT' that has the value of 'BOUGHT_SUBCAT' shifted by one month
    df_ml['TARGET'] = grouped['TARGET'].shift(-1).fillna(2)
    df_ml['TARGET'] = df_ml['TARGET'].astype(int)
    df_ml['TARGET'] = df_ml['TARGET'].replace(2, None)
    
    return df_ml


def fill_missing_values(df):

    df_cust = df.copy()

    df_cust.drop_duplicates(subset='CUSTOMER_ACCOUNT_NR_MASK', keep='first', inplace=True)

    ################### GENDER MISSING VALUES ###################

    #replace Gender values for integers
    df_cust['GENDER'].replace('M', 0,inplace=True)
    df_cust['GENDER'].replace('F', 1,inplace=True)
    #replacing nan values for 999 just to run the model
    df_cust[['FAMILY_MEMBERS']] = df_cust[['FAMILY_MEMBERS']].fillna(value=999)
    df_cust = df_cust.drop(columns=['TIME_KEY'])

    #splitting null values and not null values
    testing_data = df_cust[df_cust['GENDER'].isnull()]
    training_data = df_cust[df_cust['GENDER'].notnull()]

    X_train = training_data.drop(['GENDER'], axis=1)
    y_train = training_data['GENDER']

    knn = KNeighborsClassifier(n_neighbors=130)
    knn.fit(X_train, y_train)

    # Predict the missing values in the 'GENDER' column
    X_test = testing_data.drop(['GENDER'], axis=1)
    predicted_gender = knn.predict(X_test)

    # Assign the predicted values back to the dataframe
    df_cust.loc[df_cust['GENDER'].isnull(), 'GENDER'] = predicted_gender

    ################### FAMILY MEMBERS MISSING VALUES ###################
    df_cust['FAMILY_MEMBERS'].replace(999, np.nan, inplace=True)

    df_cust.loc[df_cust['FAMILY_MEMBERS'] > 8, 'FAMILY_MEMBERS'] = np.nan

    #Bin selection
    df_cust.loc[df_cust['FAMILY_MEMBERS'].between(0, 0, 'both'), 'FAMILY_MEMBERS_bin'] = 0
    df_cust.loc[df_cust['FAMILY_MEMBERS'].between(0, 2, 'right'), 'FAMILY_MEMBERS_bin'] = 1
    df_cust.loc[df_cust['FAMILY_MEMBERS'].between(2, 8, 'right'), 'FAMILY_MEMBERS_bin'] = 2

    testing_data = df_cust[df_cust['FAMILY_MEMBERS_bin'].isnull()]
    training_data = df_cust[df_cust['FAMILY_MEMBERS_bin'].notnull()]

    X_train = training_data.drop(['FAMILY_MEMBERS','FAMILY_MEMBERS_bin'], axis=1)
    y_train = training_data['FAMILY_MEMBERS_bin']

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=21)

    #Fit our Knn
    knn = KNeighborsClassifier(n_neighbors=38)
    knn.fit(X_train, y_train)

    # Predict the missing values in the 'GENDER' column
    X_test = testing_data.drop(['FAMILY_MEMBERS_bin', 'FAMILY_MEMBERS'], axis=1)
    predicted_family_members = knn.predict(X_test)

    # Assign the predicted values back to the dataframe
    df_cust.loc[df_cust['FAMILY_MEMBERS_bin'].isnull(), 'FAMILY_MEMBERS_bin'] = predicted_family_members

    df_cust = df_cust.drop(['FAMILY_MEMBERS'], axis=1)

    df_cust = df_cust.rename({'FAMILY_MEMBERS_bin': 'FAMILY_MEMBERS'}, axis=1)

    #replace Gender values for integers
    df_cust['GENDER'].replace(0, 'M',inplace=True)
    df_cust['GENDER'].replace(1, 'F',inplace=True)

    #replace Gender values for integers
    df_cust['FAMILY_MEMBERS'].replace(0, '(0, 0)',inplace=True)
    df_cust['FAMILY_MEMBERS'].replace(1, '(1, 2)',inplace=True)
    df_cust['FAMILY_MEMBERS'].replace(2, '(3, 8)',inplace=True)

    return df_cust
