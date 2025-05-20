# VERİ SETİ FİLTRELEME
import pandas as pd
allowed_cuisines = ["southern_us", "spanish", "italian", "french", "japanese", "turkish"]
train_df = pd.read_json("veri_seti.json")
train_filtered = train_df[train_df["cuisine"].isin(allowed_cuisines)]
train_filtered.to_json("train.json", orient="records", indent=2)
print("✅ Filtreleme tamamlandı!")
