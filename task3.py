import pandas as pd

# Sample data input: create a DataFrame to simulate raw data
data = {
    'timestamp': ['2024-05-24 08:00:00', '2024-05-24 08:30:00', '2024-05-24 09:00:00', '2024-05-25 10:00:00'],
    'activity': ['inside picking', 'outside placing', 'inside placing', 'outside picking'],
    'duration': [30, 45, 20, 60]  # durations in minutes for simplicity
}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract date from timestamp
df['date'] = df['timestamp'].dt.date

# Calculate date-wise total duration for each inside and outside
inside_durations = df[df['activity'].str.contains('inside')].groupby('date')['duration'].sum().reset_index(name='total_inside_duration')
outside_durations = df[df['activity'].str.contains('outside')].groupby('date')['duration'].sum().reset_index(name='total_outside_duration')

# Calculate date-wise number of picking and placing activities
picking_activities = df[df['activity'].str.contains('picking')].groupby('date').size().reset_index(name='picking_count')
placing_activities = df[df['activity'].str.contains('placing')].groupby('date').size().reset_index(name='placing_count')

# Merge the results into a single DataFrame
result = pd.merge(inside_durations, outside_durations, on='date', how='outer')
result = pd.merge(result, picking_activities, on='date', how='outer')
result = pd.merge(result, placing_activities, on='date', how='outer')

# Fill NaN values with 0 (optional, depends on data handling preference)
result = result.fillna(0)

# Convert the result to integer if needed
result[['total_inside_duration', 'total_outside_duration', 'picking_count', 'placing_count']] = result[['total_inside_duration', 'total_outside_duration', 'picking_count', 'placing_count']].astype(int)

# Display the result
print(result)
