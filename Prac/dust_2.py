import pandas as pd
import os

# Set directories where CSV files are located
csv_dirs = ['./_data/AIFac_pollution/TRAIN', './_data/AIFac_pollution/TEST_INPUT',
            './_data/AIFac_pollution/TRAIN_AWS', './_data/AIFac_pollution/TEST_AWS',
            './_data/AIFac_pollution/META']

# Loop through all CSV files in directories
for csv_dir in csv_dirs:
    # Create empty list to hold dataframes
    df_list = []
    
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            # Read CSV file into dataframe
            df = pd.read_csv(os.path.join(csv_dir, file))
            
            # Do any necessary data cleaning or manipulation here
            # ...
            
            # Append dataframe to list
            df_list.append(df)
    
    # Concatenate all dataframes in list into one dataframe
    combined_df = pd.concat(df_list)
    
    # Save combined dataframe to CSV file
    if csv_dir == './_data/AIFac_pollution/TRAIN':
        combined_df.to_csv('./_data/AIFac_pollution/train_all.csv', index=False)
    elif csv_dir == './_data/AIFac_pollution/TEST_INPUT':
        combined_df.to_csv('./_data/AIFac_pollution/test_all.csv', index=False)
    elif csv_dir == './_data/AIFac_pollution/TRAIN_AWS':
        combined_df.to_csv('./_data/AIFac_pollution/train_aws_all.csv', index=False)
    elif csv_dir == './_data/AIFac_pollution/TEST_AWS':
        combined_df.to_csv('./_data/AIFac_pollution/test_aws_all.csv', index=False)
    elif csv_dir == './_data/AIFac_pollution/META':
        combined_df.to_csv('./_data/AIFac_pollution/meta_all.csv', index=False)            











