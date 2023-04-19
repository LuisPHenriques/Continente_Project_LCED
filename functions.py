import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Function to format y-axis labels in millions of euros
def millions_formatter(x):
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