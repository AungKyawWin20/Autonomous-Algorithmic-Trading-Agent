import pandas as pd
import os
from pathlib import Path

# columns we want
ohlcv_columns = {"adjopen", "adjhigh", "adjlow", "adjclose", "adjvolume", "date"}

def clean_csv(file_path):
    
    df = pd.read_csv(file_path)
    df = df.rename(columns=lambda x: x.strip().lower())
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    #filter data for a specific range
    df = df[df["date"] >= "2015-01-01"]
    df = df[df["date"] <= df["date"].max()]
    df.dropna(inplace=True)

    return df

def main():
    directory = Path(__file__).parent.parent / "data" / "raw"
    cleaned_data = {} 

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = clean_csv(file_path)
            
            if ohlcv_columns.issubset(df.columns):
                cleaned_data[filename] = df
            else:
                print(f"File {filename} doesn't have the needed columns")
                
    # save in a new directory
    output_directory = Path(__file__).parent.parent / "data" / "processed"
    os.makedirs(output_directory, exist_ok=True)

    for file_name, df in cleaned_data.items():
        output_file = os.path.join(output_directory, f"{file_name}")
        df.to_csv(output_file, index=False)

    print("All cleaned files have been saved successfully!")

if __name__ == "__main__":
    main()
    
    