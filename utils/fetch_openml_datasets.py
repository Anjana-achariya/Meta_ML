from utils.openml_client import OpenMLClient

def fetch_many_datasets(n_datasets: int = 100):
    client = OpenMLClient()

    datasets = client.list_datasets(
        limit=n_datasets * 2,  # oversample because some will fail
        task_type="classification"
    )

    success = 0

    for d in datasets:
        did = int(d["did"])
        print(f"Fetching dataset {did} ...")

        ok = client.fetch_dataset(did)

        if ok:
            success += 1

        if success >= n_datasets:
            break

    print(f"\n✅ Successfully fetched {success} datasets")

if __name__ == "__main__":
    fetch_many_datasets(100)
