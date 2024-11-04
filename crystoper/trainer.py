
from sklearn.model_selection import train_test_split

def train_test_val_toy_split(df, test_size, val_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df =  train_test_split(train_df, test_size=val_size, random_state=42)
    toy_df = train_df.iloc[:10]
    
    return train_df, test_df, val_df, toy_df