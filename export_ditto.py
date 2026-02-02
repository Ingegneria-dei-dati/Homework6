import argparse
import pandas as pd


# Colonne che vogliamo serializzare (stesse per tutti, così è confrontabile)
FIELDS = ["make", "model", "year", "price", "mileage", "state", "fuel_type", "transmission"]


def serialize_record(row):
    parts = []
    for c in FIELDS:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        parts.append(f"[COL] {c} [VAL] {v}")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description="Esporta pairs CSV in formato Ditto: left\\t right\\t label")
    p.add_argument("--records", required=True, help="CSV records senza VIN (con record_id)")
    p.add_argument("--pairs", required=True, help="CSV coppie id_l,id_r,label")
    p.add_argument("--out", required=True, help="Output .txt per Ditto")
    args = p.parse_args()

    records = pd.read_csv(args.records, low_memory=False).set_index("record_id")
    pairs = pd.read_csv(args.pairs)

    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in pairs.iterrows():
            left = serialize_record(records.loc[r["id_l"]])
            right = serialize_record(records.loc[r["id_r"]])
            label = int(r["label"])
            f.write(f"{left}\t{right}\t{label}\n")

    print("Creato:", args.out)


if __name__ == "__main__":
    main()
