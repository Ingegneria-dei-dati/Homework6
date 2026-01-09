import pandas as pd
import numpy as np

# ============================================================
# 1. DEFINIZIONE DELLO SCHEMA MEDIATO
# ============================================================

MEDIATED_SCHEMA = [
    "Vin",
    "listing_id",
    "make",
    "model",
    "year",
    "price",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "drive",
    "condition",
    "color",
    "engine_cylinders",
    "latitude",
    "longitude",
    "state",
    "description",
    "listing_date",
    "source"
]

# ============================================================
# 2. ALLINEAMENTO CRAIGSLIST
# ============================================================

def align_craigslist(df: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.DataFrame()

    aligned["Vin"] = df["VIN"]
    aligned["listing_id"] = df["id"]

    aligned["make"] = df["manufacturer"]
    aligned["model"] = df["model"]
    aligned["year"] = df["year"]
    aligned["price"] = df["price"]
    aligned["mileage"] = df["odometer"]

    aligned["fuel_type"] = df["fuel"]
    aligned["transmission"] = df["transmission"]
    aligned["body_type"] = df["type"]
    aligned["drive"] = df["drive"]
    aligned["condition"] = df["condition"]
    aligned["color"] = df["paint_color"]
    aligned["engine_cylinders"] = df["cylinders"]

    aligned["latitude"] = df["lat"]
    aligned["longitude"] = df["long"]
    aligned["state"] = df["state"]

    aligned["description"] = df["description"]

    # FIX: timezone-safe
    aligned["listing_date"] = pd.to_datetime(
        df["posting_date"], errors="coerce", utc=True
    )

    aligned["source"] = "craigslist"

    return aligned[MEDIATED_SCHEMA]

# ============================================================
# 3. ALLINEAMENTO US USED CARS
# ============================================================

def align_used_cars(df: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.DataFrame()

    aligned["Vin"] = df["vin"]
    aligned["listing_id"] = df["listing_id"]

    aligned["make"] = df["make_name"]
    aligned["model"] = df["model_name"]
    aligned["year"] = df["year"]
    aligned["price"] = df["price"]
    aligned["mileage"] = df["mileage"]

    aligned["fuel_type"] = df["fuel_type"]
    aligned["transmission"] = df["transmission_display"]
    aligned["body_type"] = df["body_type"]
    aligned["drive"] = df["wheel_system_display"]

    # FIX: mappatura sicura boolean → string
    aligned["condition"] = df["is_new"].map({
        True: "new",
        False: "used"
    })

    aligned["color"] = df["exterior_color"]
    aligned["engine_cylinders"] = df["engine_cylinders"]

    aligned["latitude"] = df["latitude"]
    aligned["longitude"] = df["longitude"]

    aligned["state"] = np.nan

    aligned["description"] = df["description"]
    aligned["listing_date"] = pd.to_datetime(
        df["listed_date"], errors="coerce", utc=True
    )

    aligned["source"] = "us_used_cars"

    return aligned[MEDIATED_SCHEMA]

# ============================================================
# 4. PIPELINE DI INTEGRAZIONE
# ============================================================

def main():
    print("Caricamento dataset...")

    craigslist_df = pd.read_csv("dataset/craigslist_vehicles.csv")
    used_cars_df = pd.read_csv("dataset/used_cars_data.csv")

    print(f"Craigslist: {craigslist_df.shape}")
    print(f"Used Cars: {used_cars_df.shape}")

    print("\nAllineamento allo schema mediato...")
    craigslist_aligned = align_craigslist(craigslist_df)
    used_cars_aligned = align_used_cars(used_cars_df)

    print("Integrazione delle sorgenti...")
    integrated_df = pd.concat(
        [craigslist_aligned, used_cars_aligned],
        ignore_index=True
    )

    print("\nSchema mediato finale:")
    print(integrated_df.info())

    integrated_df.to_csv("dataset/integrated_cars.csv", index=False)
    print("\nDataset integrato salvato in dataset/integrated_cars.csv")

# ============================================================
# 5. AVVIO SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
