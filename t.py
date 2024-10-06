import pandas as pd

# Data from the table
data = {
    'Training': ['LIVE', 'LIVE', 'CSIQ', 'CSIQ', 'TID2013', 'TID2013', 'LIVEC', 'LIVEC', 'BID', 'BID', 'KonIQ', 'KonIQ'],
    'Testing': ['CSIQ', 'TID2013', 'LIVE', 'TID2013', 'LIVE', 'CSIQ', 'BID', 'KonIQ', 'LIVEC', 'KonIQ', 'LIVEC', 'BID'],
    'PQR [45]': [0.717, 0.551, 0.930, 0.546, 0.891, 0.632, 0.714, 0.757, 0.680, 0.636, 0.770, 0.755],
    'DBCNN [26]': [0.762, 0.536, 0.871, 0.523, 0.872, 0.703, 0.762, 0.754, 0.725, 0.724, 0.755, 0.816],
    'HyperIQA [27]': [0.744, None, 0.926, None, None, None, 0.756, 0.772, 0.770, 0.688, 0.785, 0.819],
    'MMMNet [28]': [0.793, 0.546, 0.890, 0.522, 0.853, 0.702, None, None, None, None, None, None],
    'VCRNet [29]': [0.768, 0.502, 0.886, 0.542, 0.822, 0.721, None, None, None, None, None, None],
    'Proposed': [0.759, 0.565, 0.938, 0.571, 0.883, 0.734, 0.879, 0.734, 0.779, 0.690, 0.791, 0.813]
}

# Create the dataframe
df = pd.DataFrame(data)

# Save to Excel file
file_path = 'cross_database_performance_comparison.xlsx'
df.to_excel(file_path, index=False)

file_path
