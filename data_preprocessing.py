import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    column_names = [
        # Add column names based on your dataset
        "duration", "protocol_type", "service", "flag", "src_bytes", 
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", 
        "su_attempted", "num_root", "num_file_creations", "num_shells", 
        "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", 
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
        "label"
    ]
    data = pd.read_csv(filepath, names=column_names)
    return data

def preprocess_data(data):
    # Encode categorical features
    label_encoders = {}
    for column in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

if __name__ == "__main__":
    data = load_data('data/kddcup.data_10_percent.gz')
    processed_data, encoders = preprocess_data(data)
    processed_data.to_csv('data/processed_data.csv', index=False)
