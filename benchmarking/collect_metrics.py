import pandas as pd

def split_metrics():
    df = pd.read_csv("data/performance_labels.csv")

    perf = df[["dataset", "model", "accuracy", "f1_score"]]
    time_df = df[["dataset", "model", "training_time"]]

    perf.to_csv("data/performance_labels.csv", index=False)
    time_df.to_csv("data/training_time_data.csv", index=False)

    print("Separated performance and time metrics")


if __name__ == "__main__":
    split_metrics()
