# Import packages for data manipulation

import pandas as pd
import numpy as np


# Import packages for data visualization

import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# Display and examine the first few rows of the dataframe

data.head()

# Get the size of the data

data.size

# Get the shape of the data

data.shape

# Get basic information about the data

data.info()

# Checking for the number of null values present in the dataset
data.isna().sum(axis=0)

# Generate a table of descriptive statistics
data.describe()

# Create a boxplot to visualize distribution of `video_duration_sec`

plt.figure(figsize=(5,1))
plt.title('Video Duration in Seconds Box Plot')
sns.boxplot(data = data, x = 'video_duration_sec')


# Create a histogram of the values in the `video_duration_sec` column to further explore the distribution of this variable.

plt.figure(figsize=(5,3))
plt.title('Video Duration in Second Histogram')
sns.histplot(data=data, x='video_duration_sec', bins = range(0,61,5))

# Create a boxplot to visualize distribution of `video_view_count`

plt.figure(figsize=(5,1))
plt.title('Video View Count Box Plot')
sns.boxplot(data['video_view_count'])


# Create a histogram of the values in the `video_view_count` column to further explore the distribution of this variable.

plt.figure(figsize=(7,3))
plt.title('Video View Count Histogram')
sns.histplot(data['video_view_count'], bins = range(0,(10**6+1), 10**5))


# Create a boxplot to visualize distribution of `video_like_count`

plt.figure(figsize=(10,1))
plt.title('Video Like Count Box Plot')
sns.boxplot(data['video_like_count'])


# Create a histogram of the values in the `video_like_count` column to further explore the distribution of this variable.

plt.figure(figsize=(7,3))
plt.title('Video Like Count Histogram')
sns.histplot(data['video_like_count'], bins = range (0,(7*10**5+1),10**5))

## making the labels on the x-axis easier to read

labels = [0] + [str(i) + 'k' for i in range(100,701,100)]
plt.xticks(range(0,7*10**5+1,10**5), labels = labels)

# Create a boxplot to visualize distribution of `video_comment_count`

plt.figure(figsize=(10,1))
plt.title('Video Comment Count Box Plot')
sns.boxplot(data['video_comment_count'])


# Create a histogram of the values in the `video_comment_count` column to further explore the distribution of this variable.

plt.figure(figsize=(7,3))
plt.title('Video Comment Count Histogram')
sns.histplot(data['video_comment_count'], bins = range(0,3001,100))


# Create a boxplot to visualize distribution of `video_share_count`

plt.figure(figsize=(10,1))
plt.title('Video Share Count Box Plot')
sns.boxplot(data['video_share_count'])


# Create a histogram of the values in the `video_share_count` column to further explore the distribution of this variable.


plt.figure(figsize=(7,3))
plt.title('Video Share Count Histogram')
sns.histplot(data['video_share_count'], bins = range(0, 300001, 10000 ))


# Create a boxplot to visualize distribution of `video_download_count`

plt.figure(figsize=(10,1))
plt.title('Video Download Count Box Plot')
sns.boxplot(data['video_download_count'])


# Create a histogram of the values in the `video_download_count` column to further explore the distribution of this variable.


plt.figure(figsize=(7,3))
plt.title('Video Download Count Histogram')
sns.histplot(data['video_download_count'], bins = range(0, 16001, 1000))


# Create a histogram with four bars: one for each combination of claim status and verification status.

plt.figure(figsize=(7,3))
plt.title('Claim Status by Verification Status Histogram')
sns.histplot(data = data, x = 'claim_status', hue = 'verified_status', multiple = 'dodge', shrink=0.5)

# Create a histogram to examine the count of each claim status for each author ban status.

plt.figure(figsize=(7,3))
plt.title('Claim Status by Author Ban Status Histogram')
sns.histplot(data=data, x = 'claim_status', hue = 'author_ban_status',
             hue_order = ['active','under review', 'banned'], palette = {'active': 'green', 'under review':'orange',
                                                                         'banned':'red'},
             multiple = 'dodge', shrink = 0.5, alpha = 0.5)


# Create a bar plot to communicate the median number of views for all videos for each seperate author ban status.


BanStatus_per_AuthorStatus = data.groupby(['author_ban_status']).median(numeric_only=True).reset_index()


plt.figure(figsize=(7,3))
plt.title('Median Number of Views per Author Ban Status')
sns.barplot(data=BanStatus_per_AuthorStatus, x = 'author_ban_status', y = 'video_view_count', 
            order = ['active','under review','banned'], palette = {'active':'green', 'under review':'orange',
                                                                   'banned':'red'},
           alpha = 0.5)


# Calculate the median view count for claim status.

data.groupby('claim_status')['video_view_count'].median()


# Create a pie graph that depicts the proportions of total views for claim videos and total views for opinion videos.

plt.figure(figsize=(4,4))
plt.title('Total Views for Claim & Opinion Videos')
plt.pie(data.groupby('claim_status')['video_view_count'].sum(), labels = ['claim','opinion'])


# Write a for loop that iterates over the column names of each count variable. For each iteration:
# 1. Calculate the IQR of the column
# 2. Calculate the median of the column
# 3. Calculate the outlier threshold (median + 1.5 * IQR)
# 4. Calculate the numer of videos with a count in that column that exceeds the outlier threshold
# 5. Print "Number of outliers, {column name}: {outlier count}"
# 
# ```
# Example:
# Number of outliers, video_view_count: ___
# Number of outliers, video_like_count: ___
# Number of outliers, video_share_count: ___
# Number of outliers, video_download_count: ___
# Number of outliers, video_comment_count: ___
# ```

column_names = ['video_view_count',
              'video_like_count',
              'video_share_count',
              'video_download_count',
              'video_comment_count',
              ]

for column in column_names:
    quarter3 = data[column].quantile(0.75)
    quarter1 = data[column].quantile(0.25)
    iqr = quarter3-quarter1
    median = data[column].median()
    outliers = median + 1.5*iqr
    
    outlier_amount = (data[column] > outliers).sum()
    print(f'Number of outliers, {column}:',outlier_amount)



# Create a scatterplot of `video_like_count` versus `video_comment_count` according to 'claim_status'

sns.scatterplot(x=data['video_like_count'], y=data['video_comment_count'], hue = data['claim_status'], s=10)


# Create a scatterplot of `video_like_count` versus `video_comment_count` for opinions only

opinion_data = data[data['claim_status']=='opinion']

sns.scatterplot(x=opinion_data['video_like_count'], y=opinion_data['video_comment_count'], s=10)
