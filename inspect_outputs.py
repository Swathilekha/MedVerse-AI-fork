import pandas as pd

files = [
    "data/clinical_text/fusion_metrics.csv",
    "data/clinical_text/fusion_conflict_analysis.csv",
    "data/clinical_text/fusion_calibration.csv",
    "data/clinical_text/multimodal_fusion_results.csv"
]

for file in files:
    print("\n" + "=" * 90)
    print("FILE:", file)
    print("=" * 90)

    try:
        df = pd.read_csv(file)

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

    except Exception as e:
        print("ERROR:", e)