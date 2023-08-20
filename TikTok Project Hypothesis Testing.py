
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

# Understanding the dataset
data.head()

data.describe(include = 'all')

# Check for missing data
data.isnull().sum()

# Drop rows with missing data
data = data.dropna(axis = 0)

# Recheck the dataset after dropping missing data
data.head()

# Finding the mean video view count for each verification status
data.groupby('verified_status')['video_view_count'].mean()

# Visualizing the mean video view count for each verification status
plt.figure(figsize = (7,5))
plt.title('Video View Count Average by Verified Status')
sns.barplot(data = data, x = 'verified_status', y = 'video_view_count')

# Hypothesis testing
significance_level = 0.05

verified = data[data['verified_status']== 'verified']['video_view_count']
not_verified = data[data['verified_status']== 'not verified']['video_view_count']

stats.ttest_ind(a = not_verified, b = verified, equal_var = False)


#   Based on the extrememly small P value we reject the null hypothesis and state that there IS a statistically significant difference between the mean video view counts between verified and unverified users on TikTok
