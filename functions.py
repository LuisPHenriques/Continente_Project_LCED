import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
        if interval != 'YEAR':
            aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', interval, 'YEAR', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', interval]) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', interval: 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', interval], how='left')
    
        else:
            aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR']) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'YEAR'], how='left')
    
    elif f_type == 'SUBCAT':
        if interval != 'YEAR':
            aggregations = df_transactions[['SUBCAT_CD_EXT', interval, 'YEAR', feature+'_'+interval]] \
                                .groupby(['SUBCAT_CD_EXT', 'YEAR', interval]) \
                                .agg({'SUBCAT_CD_EXT': 'first', interval: 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['SUBCAT_CD_EXT', 'YEAR', interval], how='left')
    
        else:
            aggregations = df_transactions[['SUBCAT_CD_EXT', 'YEAR', feature+'_'+interval]] \
                                .groupby(['SUBCAT_CD_EXT', 'YEAR']) \
                                .agg({'SUBCAT_CD_EXT': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['SUBCAT_CD_EXT', 'YEAR'], how='left')

    else:
        if interval != 'YEAR':
            aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', interval, 'YEAR', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', interval]) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', 'SUBCAT_CD_EXT': 'first', interval: 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', interval], how='left')
    
        else:
            aggregations = df_transactions[['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR', feature+'_'+interval]] \
                                .groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR']) \
                                .agg({'CUSTOMER_ACCOUNT_NR_MASK': 'first', 'SUBCAT_CD_EXT': 'first', 'YEAR': 'first', feature+'_'+interval: 'last'}) \
                                .reset_index(drop=True)

            # merge the aggregations dataframe with the new_df dataframe on the customer id, subcategory id, and month columns
            df_ml = pd.merge(df_ml, aggregations, on=['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT', 'YEAR'], how='left')

    if avg:
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].fillna(0)
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].astype(float).round(2)
    else:
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].fillna(0)
        df_ml[feature+'_'+interval] = df_ml[feature+'_'+interval].astype(int)    

    return df_ml


def compute_target(df_ml):
    # create a new column 'BOUGHT_SUBCAT' that indicates if a customer bought a subcategory in a given month
    df_ml['BOUGHT_SUBCAT'] = (df_ml['CUSTSUBCAT_NUM_TRANSACTIONS_MONTH'] > 0).astype(int)
    
    # group the dataframe by customer_id and category_id
    grouped = df_ml.groupby(['CUSTOMER_ACCOUNT_NR_MASK', 'SUBCAT_CD_EXT'])

    # create a new column 'PREV_BOUGHT_SUBCAT' that has the value of 'BOUGHT_SUBCAT' shifted by one month
    df_ml['TARGET'] = grouped['BOUGHT_SUBCAT'].shift(-1).fillna(2)
    df_ml['TARGET'] = df_ml['TARGET'].astype(int)
    df_ml['TARGET'] = df_ml['TARGET'].replace(2, None)

    df_ml = df_ml.drop(columns=['BOUGHT_SUBCAT'])
    
    return df_ml