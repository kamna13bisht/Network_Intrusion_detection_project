from data_preprocessing import preprocess_data, load_data
from model import train_model

def main():
    data = load_data('data/kddcup.data_10_percent.gz')
    processed_data, _ = preprocess_data(data)
    processed_data.to_csv('data/processed_data.csv', index=False)
    
    model, X_test, y_test = train_model(processed_data)
    # Additional code to save the model, etc.

if __name__ == "__main__":
    main()
