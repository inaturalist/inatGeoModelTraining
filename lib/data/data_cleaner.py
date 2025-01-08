def clean_dataset(data):
    num_obs = len(data)
    data = data[
        (
            (data["latitude"] <= 90)
            & (data["latitude"] >= -90)
            & (data["longitude"] <= 180)
            & (data["longitude"] >= -180)
        )
    ]
    if (num_obs - len(data)) > 0:
        print(f"  {num_obs - len(data)} items filtered due to invalid locs")

    num_obs = len(data)
    data = data.dropna()
    if (num_obs - len(data)) > 0:
        print(f"  {num_obs - len(data)} items filtered due to NaN entry")

    print(f"  after cleaning, we have {len(data)} records.")
    return data
