#%% Load libraries
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
from lifelines.utils import to_long_format, add_covariate_to_timeline
import warnings

from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("display.max_columns", None)
tqdm.pandas()

#%% Read data
df_ndr = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_NDR.csv") # Diabetes register
df_sesar_iv = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_SESAR_IV.csv") # OSA register diagnosis
df_sesar_ts = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_SESAR_TS.csv") # OSA register treatment start
df_sesar_fu = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_SESAR_FU.csv") # OSA register follow up
df_dr = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag Socialstyrelsen 2024/ut_r_dors_63851_2023.csv", index_col=0) # Cause of death register
df_pr_ip = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag Socialstyrelsen 2024/ut_r_par_sv_63851_2023.csv", index_col=0) # Patient register (in-patient)

#%% Helper functions
def fix_b_cols(b_date_string):
    # Some date columns have strings in the format "b'20000904'", extract numbers only
    if isinstance(b_date_string, str):
      return b_date_string.replace("b'", "").replace("'", "")
    else:
       return b_date_string


def fix_00_days(date_string):
  # Soms date strings in DORS have unspecified day or month denoted by 00, which doesn't allow for conversion to datetime
  if date_string.endswith("00"):
    date_list = list(date_string)
    date_list[-2:] = "15"
    date_string = "".join(date_list)
  
  return date_string


def fix_greg_dates(greg_date):
    # SESAR has dates in the form of number of hours after 14 okt. 1582
    if isinstance(greg_date, str):
        orig_time = datetime(1582, 10, 14)
        return orig_time + timedelta(hours=int(re.sub(":.*", "", greg_date)))
    else:
        return greg_date


#%% Fix unspecified death dates in DORS
df_dr = df_dr[df_dr.LopNr.notnull()] # Filter out rows without person indentifier
df_dr = df_dr.assign(DODSDAT=df_dr.DODSDAT.apply(fix_b_cols).apply(fix_00_days)) # Set 00 days to 15
set_unspecified_death_month = set(df_dr[df_dr.DODSDAT.str[4:6] == "00"].LopNr)
df_dr = df_dr[~df_dr.LopNr.isin(set_unspecified_death_month)] # Remove 00 months from data (~1300 rows)
df_dr = df_dr.assign(DODSDAT=pd.to_datetime(df_dr.copy().DODSDAT, format="%Y%m%d")) # Change to datetime

#%% Filter out rows without person identifier and unspecified death dates
df_t2d = df_ndr[df_ndr.LopNr.notnull() & (df_ndr.klin_diab_typ == 2) & ~df_ndr.LopNr.isin(set_unspecified_death_month)] # Also keep only T2D
df_sesar_iv = df_sesar_iv[df_sesar_iv.LopNr.notnull() & ~df_sesar_iv.LopNr.isin(set_unspecified_death_month)]
df_sesar_ts = df_sesar_ts[df_sesar_ts.LopNr.notnull() & ~df_sesar_ts.LopNr.isin(set_unspecified_death_month)]
df_sesar_fu = df_sesar_fu[df_sesar_fu.LopNr.notnull() & ~df_sesar_fu.LopNr.isin(set_unspecified_death_month)]
df_dr = df_dr[df_dr.LopNr.notnull()]
df_pr_ip = df_pr_ip[df_pr_ip.LopNr.notnull() & df_pr_ip.INDATUMA.notnull() & ~df_pr_ip.LopNr.isin(set_unspecified_death_month)] # Also remove 1 row without check-in date

# Fix important datetime columns
df_t2d = df_t2d.assign(regdat=pd.to_datetime(df_t2d.regdat))
df_sesar_iv = df_sesar_iv.assign(IV_DiagnosisDate=pd.to_datetime(df_sesar_iv.IV_DiagnosisDate))
df_sesar_ts = df_sesar_ts.assign(TS_CPAPStart=df_sesar_ts.TS_CPAPStart.apply(fix_greg_dates))
df_sesar_fu["FU_FollowUpDate"] = pd.to_datetime(df_sesar_fu.FU_FollowUpDate, format="%Y-%m-%d")
df_sesar_fu["FU_CPAPMachineUsage"] = pd.to_datetime(df_sesar_fu.FU_CPAPMachineUsage, format="%H:%M:%S").apply(lambda time: timedelta(hours=time.hour, minutes=time.minute, seconds=time.second) if pd.notna(time) else pd.NaT)
df_sesar_fu["FU_CPAPMachineUsage2"] = pd.to_datetime(df_sesar_fu.FU_CPAPMachineUsage2, format="%H:%M:%S").apply(lambda time: timedelta(hours=time.hour, minutes=time.minute, seconds=time.second) if pd.notna(time) else pd.NaT)
df_sesar_fu["FU_CPAPEndDate"] = df_sesar_fu.FU_CPAPEndDate.apply(fix_greg_dates)

#%% Identify MACE events from patient register
# Fix the "b'...'" columns
dia_col = [col for col in df_pr_ip if (col.startswith("DIA")) & (col != "DIA_ANT")] 
b_col = dia_col + ["hdia", "SJUKHUS", "INDATUMA", "UTDATUMA", "PVARD"]
df_pr_ip[b_col] = df_pr_ip[b_col].map(fix_b_cols)
df_pr_ip = df_pr_ip.assign(INDATUMA=df_pr_ip.INDATUMA.replace("20108024", "20100824")) # Replace a single whacky value
df_pr_ip = df_pr_ip.assign(INDATUMA=pd.to_datetime(df_pr_ip.INDATUMA, format="%Y%m%d")) # To datetime

# Define MACE event codes
mace_codes = {
    'ami': r'^I21|^I22',
    'stroke': r'^I63|^I65|^I66',
    # 'heart_failure': r'^I11.0|^I50|^I97.1', # Don't consider this for now
    # 'cv_death': r'^I46.1|^I46.9' # Not in data, ignore 
}

# Apply MACE event detection per patient
for event, pattern in mace_codes.items():
    df_pr_ip[event] = df_pr_ip[dia_col].apply(lambda col: col.str.contains(pattern, na=False)).any(axis=1).astype(int)

# Create composite MACE variable
df_pr_ip['mace_composite'] = df_pr_ip[list(mace_codes.keys())].any(axis=1).astype(int)

# Keep only rows in which a MACE event occurred
df_pr_ip_mace = df_pr_ip[df_pr_ip.mace_composite == 1]

#%% Define some commonly used subsets
df_t2d_diag = df_t2d.sort_values(["LopNr", "regdat"]).drop_duplicates("LopNr", keep="first") # Define T2D diagnosis date as first regdat
df_osa_diag = df_sesar_iv.sort_values(["LopNr", "IV_DiagnosisDate"]).drop_duplicates("LopNr", keep="first") # Define OSA diagnosis as first diagnosis date in case of duplicates
df_cpap_start = df_sesar_ts[df_sesar_ts.TS_CPAPStart.notnull()].sort_values(["LopNr", "TS_CPAPStart"]).drop_duplicates("LopNr", keep="first") # Keep first CPAP start
df_cpap_end = df_sesar_fu[df_sesar_fu.FU_CPAPEndDate.notnull()].sort_values(["LopNr", "FU_CPAPEndDate"]).drop_duplicates("LopNr", keep="last") # Keep last end date

#%% Main functions
def find_best_year(row):
    diag_year = row["diag_year"]
    matched_years = sorted(row["matched_years"])

    if len(matched_years) == 0:
        return np.nan
    elif diag_year in matched_years:
        # Person has a record in their diagnosis year
        return diag_year
    else:
        # Compute distances of other matched years
        year_dist = [abs(year - diag_year) for year in matched_years]
        
        # Return year closest to diagnosis, preferencing before diagnosis
        return matched_years[year_dist.index(min(year_dist))]


def match_people_to_lisa_years(df_subset, diag_col="regdat"):
    # Start dataframe with LopNr and diagnosis year
    df_ppl_year_match = pd.concat([df_subset.LopNr, df_subset[diag_col].dt.year], axis=1).rename(columns={diag_col: "diag_year"})
    df_ppl_year_match["matched_years"] = [[] for _ in range(len(df_ppl_year_match.index))]

    # Iterate over diagnosis years to check if people appear in LISA
    for year in df_ppl_year_match.diag_year.unique():
        if year == 2022:
            # 2022 misses important columns
            continue

        try:
            # Read LISA data from year and save year to person if matched
            df_lisa_year = pd.read_csv(f"/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_LISA_{year}.csv")
            df_ppl_year_match.loc[df_ppl_year_match.LopNr.isin(df_lisa_year.LopNr), "matched_years"] = df_ppl_year_match.loc[df_ppl_year_match.LopNr.isin(df_lisa_year.LopNr), "matched_years"].apply(lambda matched_years: matched_years + [year])
        except:
            # If year doesn't have data, skip
            continue

    # Find the best year based on distance to diagnosis date
    df_ppl_year_match["best_year"] = df_ppl_year_match.apply(find_best_year, axis=1)

    return df_ppl_year_match


def merge_socio_economic_data(df_subset, soc_ec_var, diag_col="regdat"):
    # Get birth country
    df_birth_place = pd.read_csv("/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_Grunduppgifter.csv")
    df_birth_place = df_birth_place[~df_birth_place.FodelseLand_EU27_2020.isin(["Statslös", "Okänt"]) & df_birth_place.FodelseLand_EU27_2020.notnull()]
    df_subset_w_soc_ec = df_subset.merge(df_birth_place[["LopNr", "FodelseLand_EU27_2020"]], how="left", on="LopNr")
    df_subset_w_soc_ec["FodelseLand_EU27_2020"] = df_subset_w_soc_ec["FodelseLand_EU27_2020"].apply(lambda birth_place: 1 if birth_place == "Sverige" else 0)

    # Define soc-ec columns to add and initialize empty values
    df_subset_w_soc_ec[soc_ec_var] = np.nan
    df_subset_w_soc_ec.set_index("LopNr", inplace=True)

    # Retrieve matched LISA years
    df_ppl_year_match = match_people_to_lisa_years(df_subset, diag_col)

    # Iterate over matched years
    for year in df_ppl_year_match.best_year.unique():
        if np.isnan(year):
            continue

        year = int(year)
        print(f"Processing {year}...")

        # Load LISA data of this year and filter for people in NDR
        df_lisa_year = pd.read_csv(f"/vault/marja987_amed/NDR-SESAR/Uttag SCB+NDR+SESAR 2024/FI_Lev_LISA_{year}.csv")
        lopnr_year = df_ppl_year_match.loc[df_ppl_year_match.best_year == int(year), "LopNr"]
        df_lisa_year = df_lisa_year.loc[df_lisa_year.LopNr.isin(lopnr_year)].drop_duplicates("LopNr")

        try:
            # Different education level column name in some years
            df_lisa_year = df_lisa_year[[col for col in soc_ec_var if col != "Sun2000niva_old"] + ["Sun2020Niva_Old", "LopNr"]].rename(columns={"Sun2020Niva_Old": "Sun2000niva_old"})
        except KeyError:
            try:
                df_lisa_year = df_lisa_year[soc_ec_var + ["LopNr"]]
            except KeyError:
                df_lisa_year = df_lisa_year[[col for col in soc_ec_var if col != "Sun2000niva_old"] + ["Sun2000niva_Old", "LopNr"]].rename(columns={"Sun2000niva_Old": "Sun2000niva_old"})

        df_lisa_year.set_index("LopNr", inplace=True)
        df_subset_w_soc_ec.update(df_lisa_year)

        print(f"{year} done\n")

    df_subset_w_soc_ec.reset_index(inplace=True)

    # Turn civil status into widowed binary
    print("Percentage of Civil missing:", df_subset_w_soc_ec.Civil.isnull().sum() / len(df_subset_w_soc_ec.index))
    df_subset_w_soc_ec["Civil"] = df_subset_w_soc_ec.Civil.apply(lambda civil_stat: 1 if civil_stat == "Ä" else (0 if pd.notna(civil_stat) else np.nan))

    # Get leading category for household type
    df_subset_w_soc_ec["HushallsTyp_RTB"] = np.floor(df_subset_w_soc_ec["HushallsTyp_RTB"]).astype("Int64")
    df_subset_w_soc_ec["Sun2000niva_old"] = df_subset_w_soc_ec.Sun2000niva_old.replace("*", np.nan).astype("Int64")

    return df_subset_w_soc_ec


def impute_within_time_range(df_diag, df_long, var_set, diag_col="regdat", td=pd.Timedelta(days=365.25 / 2)):
    # Impute baseline values by seeing if measurements exist in follow-up appointments
    missing_rates = (df_diag.isnull().sum(axis=0) / len(df_diag.index)).sort_values(ascending=True)
    print(missing_rates[var_set + ["bmi"]])
    df_long = df_long.sort_values(["LopNr", diag_col])

    for var in var_set + ["bmi", "vikt", "langd"]:
        print(f"Processing {var}...")

        # Get rows in T2D diag where variable is missing
        df_missing_nr_dat = df_diag.loc[df_diag[var].isnull(), ["LopNr", diag_col]]
        df_nonmissing = df_long.loc[df_long[var].notnull(), ["LopNr", diag_col, var]]

        # See if there's a non-missing value within six months of diagnosis
        df_imputations = pd.merge_asof(df_missing_nr_dat.sort_values(diag_col), df_nonmissing.sort_values(diag_col), by="LopNr", on=diag_col, direction="nearest", tolerance=td)

        # Update diagnosis values where a match was found
        df_diag = df_diag.merge(df_imputations, how="left", on=["LopNr", diag_col], suffixes=("", "_imputed"))
        df_diag[var] = df_diag[var].fillna(df_diag[f"{var}_imputed"])
        df_diag.drop(columns=[f"{var}_imputed"], inplace=True)

        print(f"Imputed {df_imputations[var].notnull().sum()} rows\n")
        
    # Compute BMI where possible
    df_diag.loc[df_diag.bmi.isnull() & ((df_diag.vikt.notnull() & df_diag.langd.notnull())), "bmi"] = df_diag.loc[df_diag.bmi.isnull() & ((df_diag.vikt.notnull() & df_diag.langd.notnull())), "vikt"] / (df_diag.loc[df_diag.bmi.isnull() & ((df_diag.vikt.notnull() & df_diag.langd.notnull())), "langd"] / 100) ** 2

    # Print missingness rates
    missing_rates = (df_diag.isnull().sum(axis=0) / len(df_diag.index)).sort_values(ascending=True)
    print(missing_rates[var_set + ["bmi"]])

    return df_diag


def compute_drug_usage(df_diag, diag_col="regdat"):
    # Define used drugs with ATC codes
    drug_codes = {
    "antithrombotic_agents": r"^B01A", # Important for MACE prevention
    "antihypertensive_comb": r"^C02|^C08|^C09", # Combined antihypertensive drugs
    "lipid_modifying_agents": r"^C10", # Lower cholesterol and stabilize plaque
    # "antihypertensive_agents": r"^C02", # Halt progression of plaque formation (longer time frame)
    # "calcium_channel_blockers": r"^C08", # Antihypertensive
    # "angiotensin_converting_enzyme_inhibitors": r"^C09", # Antihypertensive (usually first line)
    # "antidiabetics": r"^A10",
    # "beta_blockers": r"^C07", 
    # "systemic_antibacterial_agents": r"^J01",
    # "systemic_antifungals": r"^J02",
    # "antiviral_agents_for_systemic_use": r"^J05",
    # "opioid_analgesics": r"^N02A | ^B",
    # "antidepressants": r"^N06A",
    # "hypnotics_and_sedatives": r"^N05C",
    # "diuretics": r"^C03" # Can be used as hypertensive, but almost always as add-on to another
    }   

    # Load drug register and T2D diagnosis set
    df_lmed = pl.scan_csv("/vault/marja987_amed/NDR-SESAR/Uttag Socialstyrelsen 2024/ut_r_lmed_63851_2023.csv", has_header=True, infer_schema=True)
    df_diag = pl.from_pandas(df_diag).lazy()

    # Merge dataframes and fix ATC column format
    df_merged = df_diag.select(["LopNr", diag_col]).join(df_lmed.select(["LopNr", "ATC", "EDATUM"]), on="LopNr", how="inner")
    df_merged = df_merged.with_columns(pl.col("ATC").str.replace("b'", "").str.replace("'", ""))

    # Filter out relevant medications and create binary indicators for category
    df_merged = df_merged.with_columns([pl.col("ATC").str.contains(pattern).cast(pl.Int8).fill_null(0).alias(drug) for drug, pattern in drug_codes.items()])

    # Extraction and diagnosis dates to date column type
    df_merged = df_merged.with_columns(pl.col("EDATUM").str.strptime(pl.Date, format="%Y-%m-%d")).sort(["LopNr", "EDATUM"])

    # Compute days between diagnosis and drug extraction
    df_merged = df_merged.with_columns((pl.col("EDATUM") - pl.col(diag_col)).dt.total_days().abs().alias("diff_abs"))
    df_merged = df_merged.filter((pl.col("diff_abs") <= 61))
    df_merged = df_merged.unique(["LopNr"] + [drug for drug in drug_codes.keys()], keep="first")

    # Get drug binaries and join with T2D diagnosis set
    df_max = df_merged.group_by("LopNr").agg([pl.max(drug) for drug in drug_codes.keys()])
    df_final = df_diag.join(df_max, on="LopNr", how="left")

    # Run query
    df_final = df_final.collect().to_pandas()
    df_final[list(drug_codes.keys())] = df_final[list(drug_codes.keys())].fillna(0)

    return df_final


def merge_mace_death(df_subset, diag_col="regdat", plot_missingness=True):
    print(f"No. people in subset: {df_subset.LopNr.nunique()}")
    
    # Merge the data with the in-patient and death registers
    df_subset_mace = df_subset.merge(df_pr_ip_mace[["LopNr", "INDATUMA", "stroke", "ami"]], how="left", on="LopNr", suffixes=("_data", None))
    df_subset_mace_death = df_subset_mace.merge(df_dr[["LopNr", "DODSDAT"]], how="left", on="LopNr")

    # Drop some people that have a death before their diagnosis
    n_people_before = df_subset_mace_death.LopNr.nunique()
    len_before = len(df_subset_mace_death.index)
    df_subset_mace_death = df_subset_mace_death[(df_subset_mace_death["DODSDAT"] > df_subset_mace_death[diag_col]) | df_subset_mace_death.DODSDAT.isnull()]
    n_people_after = df_subset_mace_death.LopNr.nunique()
    len_after = len(df_subset_mace_death.index)
    print(f"No. people dropped for death before diagnosis: {n_people_before - n_people_after}")
    print(f"No. rows dropped for death before diagnosis: {len_before - len_after}")

    if plot_missingness:
        # Plot missingness in final subset
        missing_rates = (df_subset_mace_death.isnull().sum(axis=0) / len(df_subset_mace_death.index)).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 18))
        ax.barh(missing_rates.index, missing_rates)
        plt.show()

    return df_subset_mace_death


def compute_duration(df_group, diag_col="regdat"):
    extraction_date = datetime(2024, 5, 15) # Data extraction date, i.e. end of studied period
    
    # Prepare group data
    lopnr = df_group.name
    df_group = df_group.sort_values("INDATUMA").reset_index(drop=True)
    diag_date = df_group.iloc[0][diag_col]
    
    # Initialize treatment-related variables
    cpap_treated = 0
    time_to_cpap_start = time_to_cpap_end = (extraction_date - diag_date).days * 1.1 # Initialize at bit more than studied time

    if not df_cpap_start[df_cpap_start.LopNr == lopnr].empty:
        # Person had CPAP treatment
        cpap_treated = 1
        start_date = df_cpap_start.loc[df_cpap_start.LopNr == lopnr, "TS_CPAPStart"].iloc[0]
        time_to_cpap_start = (start_date - diag_date).days

        if not df_cpap_end[df_cpap_end.LopNr == lopnr].empty:
            # Person has a CPAP end date
            end_date = df_cpap_end.loc[df_cpap_end.LopNr == lopnr, "FU_CPAPEndDate"].iloc[0]
            if end_date > start_date:
                # End date came after earliest logged start date
                time_to_cpap_end = (end_date - diag_date).days

    # Initialize event and time-to-event settings
    ami = stroke = death = ami_history = stroke_history = 0
    time_to_ami = time_to_stroke = time_to_death = (extraction_date - diag_date).days # Initialize at studied time, i.e. censoring time

    if not df_group[df_group.ami == 1].empty:
        # Person had AMI
        if not df_group[(df_group.ami == 1) & (df_group.INDATUMA > diag_date)].empty:
            # Person had AMI during studied period
            ami_date = df_group.loc[(df_group.ami == 1) & (df_group.INDATUMA > diag_date), "INDATUMA"].iloc[0]
            time_to_ami = (ami_date - diag_date).days
            ami = 1
        else:
            # History of AMI
            ami_history = 1
    if not df_group[df_group.stroke == 1].empty:
        # Person had a stroke
        if not df_group[(df_group.stroke == 1) & (df_group.INDATUMA > diag_date)].empty:
            # Person had stroke during studied period
            stroke_date = df_group.loc[(df_group.stroke == 1) & (df_group.INDATUMA > diag_date), "INDATUMA"].iloc[0]
            time_to_stroke = (stroke_date - diag_date).days
            stroke = 1
        else:
            # History of stroke
            stroke_history = 1
    if df_group["DODSDAT"].notnull().any():
        # Person has a death date
        death = 1
        time_to_death = (df_group["DODSDAT"].iloc[0] - diag_date).days

        # Censor at death for other events
        if ami == 0:
            time_to_ami = time_to_death
        if stroke == 0:
            time_to_stroke = time_to_death
    
    return ami, stroke, death, time_to_ami, time_to_stroke, time_to_death, ami_history, stroke_history, cpap_treated, time_to_cpap_start, time_to_cpap_end
        

def to_surv_analysis_format(df_subset, diag_col="regdat", var_set=[], plot_missingness=True):
    df_subset_mace_death = merge_mace_death(df_subset, diag_col, plot_missingness)
    
    # Compute survival times per event
    df_survival_times = df_subset_mace_death.groupby("LopNr").progress_apply(compute_duration, diag_col=diag_col, include_groups=False).reset_index()
    df_survival_times[["ami", "stroke", "death", "time_to_ami", "time_to_stroke", "time_to_death", "ami_history", "stroke_history", "cpap_treated", "time_to_cpap_start", "time_to_cpap_end"]] = pd.DataFrame(df_survival_times[0].to_list(), index=df_survival_times.index)
    df_survival_times = df_survival_times.merge(df_subset_mace_death.drop_duplicates("LopNr", keep="first")[["LopNr", "alder", "sex", "bmi"] + var_set], how="inner", on="LopNr")
    df_survival_times["sex"] -= 1
    df_survival_times.drop(0, axis=1, inplace=True)

    print(f"No. people in survival times set: {df_survival_times.LopNr.nunique()}")
    print(f"No. people in treated group: {df_survival_times[df_survival_times.cpap_treated == 1].LopNr.nunique()}")

    return df_survival_times


def to_time_varying(df_survival_times, df_longitudinal, var_set):
    # Define relevant columns
    list_events = ["death", "ami", "stroke"]
    event_cols = [event_col for event in list_events for event_col in (event, f"time_to_{event}")]
    relevant_cols = [col for col in df_survival_times.columns if col not in event_cols] + ["start", "stop", "event"]
    dict_df_events = {event: None for event in list_events}

    # Iterate over events
    for event in list_events:
        print(f"Processing {event}...")

        # Transform time to event column into start, stop, event
        df_event_tv = to_long_format(df_survival_times, duration_col=f"time_to_{event}")
        df_event_tv = df_event_tv.rename(columns={event: "event"}).drop(event_cols, axis=1, errors="ignore")
        
        ### Add in other covariates ###
        for var in var_set + ["bmi"]:
            print(f"\tProcessing {var}...")
            
            # Drop missing values in this variable and keep the first in case of multiple measurements in a day
            df_var = df_longitudinal.dropna(subset=var).drop_duplicates(subset=["LopNr", "regdat"], keep="first")

            # Drop back to back duplicate values 
            mask = (df_var.LopNr == df_var.LopNr.shift()) & (df_var[var] == df_var[var].shift())
            df_var = df_var[~mask]

            # Add covariate to time-varying data
            df_event_tv = add_covariate_to_timeline(df_event_tv, df_var[["LopNr", "time_since_diag", var]], duration_col="time_since_diag", id_col="LopNr", event_col="event")
            
            print(f"\t{var} done.\n")

        ### Transform CPAP treated variable ###
        # Sort people by time
        df_event_tv = df_event_tv[relevant_cols].sort_values(by=["LopNr", "start"])

        # Adjust the format of the treated
        for lopnr, df_person in df_event_tv[df_event_tv.cpap_treated == 1].groupby("LopNr"):
            time_to_cpap_start = df_person["time_to_cpap_start"].iloc[0]
            event_time = df_person["stop"].iloc[-1]

            if time_to_cpap_start <= 0:
                # Treatment started before study period
                # Nothing needs to change
                continue
            elif time_to_cpap_start >= event_time:
                # Treatment started after event/censoring
                # CPAP treatment = 0 from start
                df_event_tv.loc[df_event_tv.LopNr == lopnr, "cpap_treated"] = 0
            else:
                # Treatment started during studied period
                split_index = df_person.index[(df_person["start"] <= time_to_cpap_start) & (df_person["stop"] > time_to_cpap_start)]
                
                # Duplicate row and set the start to the treatment start time
                split_row = df_person.loc[split_index, :]
                split_row["start"] = time_to_cpap_start

                # Set end to treatment start time in original row, and concat back together
                df_event_tv.loc[split_index, "stop"] = time_to_cpap_start
                df_event_tv = pd.concat([df_event_tv, split_row], ignore_index=True)
                df_event_tv.loc[(df_event_tv.LopNr == lopnr) & (df_person.start < time_to_cpap_start), "cpap_treated"] = 0
                
        df_event_tv = df_event_tv.drop(["time_to_cpap_start", "time_to_cpap_end"], axis=1)
        dict_df_events[event] = df_event_tv
        print(f"{event} done.")

    return dict_df_events


def to_wide_tv_data(df_event_tv):
    df_event_tv = df_event_tv.sort_values(["LopNr", "start"])
    df_wide = df_event_tv.groupby("LopNr").last().reset_index().drop(columns=var_set + ["bmi"])

    for col in var_set + ["bmi"]:
        df_col_unique = df_event_tv.loc[df_event_tv[col] != df_event_tv.groupby("LopNr")[col].shift(), ["LopNr", col]]
        df_col_unique[f"{col}_idx"] = df_col_unique.groupby("LopNr").cumcount()
        df_col_wide = df_col_unique.pivot(index="LopNr", columns=f"{col}_idx", values=col)
        df_col_wide.columns = [f"{col}_{i}" for i in df_col_wide.columns]
        
        df_wide = df_wide.merge(df_col_wide, how="inner", left_on="LopNr", right_index=True)

    return df_wide


def train_test_impute(df_survival_times, var_set, train_size=0.75):
    # Train-test split followed by simple mean imputation
    df_train, df_test = train_test_split(df_survival_times, train_size=train_size, stratify=df_survival_times["cpap_treated"])

    print("Missing rates train:\n")
    print(df_train[var_set].isnull().sum() / len(df_train.index))
    print("Missing rates test:\n")
    print(df_test[var_set].isna().sum() / len(df_test.index))

    # Impute categorical variables by sampling
    train_rokare_counts = df_train.rokare.value_counts(normalize=True)
    df_train.loc[df_train.rokare.isna(), "rokare"] = np.random.choice(train_rokare_counts.index, len(df_train[df_train.rokare.isna()].index), p=train_rokare_counts)
    df_test.loc[df_test.rokare.isna(), "rokare"] = np.random.choice(train_rokare_counts.index, len(df_test[df_test.rokare.isna()].index), p=train_rokare_counts)

    train_edu_counts = df_train.Sun2000niva_old.value_counts(normalize=True)
    df_train.loc[df_train.Sun2000niva_old.isna(), "Sun2000niva_old"] = np.random.choice(train_edu_counts.index, len(df_train[df_train.Sun2000niva_old.isna()].index), p=train_edu_counts)
    df_test.loc[df_test.Sun2000niva_old.isna(), "Sun2000niva_old"] = np.random.choice(train_edu_counts.index, len(df_test[df_test.Sun2000niva_old.isna()].index), p=train_edu_counts)

    train_household_counts = df_train.HushallsTyp_RTB.value_counts(normalize=True)
    df_train.loc[df_train.HushallsTyp_RTB.isna(), "HushallsTyp_RTB"] = np.random.choice(train_household_counts.index, len(df_train[df_train.HushallsTyp_RTB.isna()].index), p=train_household_counts)    
    df_test.loc[df_test.HushallsTyp_RTB.isna(), "HushallsTyp_RTB"] = np.random.choice(train_household_counts.index, len(df_test[df_test.HushallsTyp_RTB.isna()].index), p=train_household_counts)

    train_widowed_counts = df_train.Civil.value_counts(normalize=True)
    df_train.loc[df_train.Civil.isna(), "Civil"] = np.random.choice(train_widowed_counts.index, len(df_train[df_train.Civil.isna()].index), p=train_widowed_counts)
    df_test.loc[df_test.Civil.isna(), "Civil"] = np.random.choice(train_widowed_counts.index, len(df_test[df_test.Civil.isna()].index), p=train_widowed_counts)

    train_sweden_counts = df_train.FodelseLand_EU27_2020.value_counts(normalize=True)
    df_train.loc[df_train.FodelseLand_EU27_2020.isna(), "FodelseLand_EU27_2020"] = np.random.choice(train_sweden_counts.index, len(df_train[df_train.FodelseLand_EU27_2020.isna()].index), p=train_sweden_counts)
    df_test.loc[df_test.FodelseLand_EU27_2020.isna(), "FodelseLand_EU27_2020"] = np.random.choice(train_sweden_counts.index, len(df_test[df_test.FodelseLand_EU27_2020.isna()].index), p=train_sweden_counts)

    # Impute continuous variables
    train_means = df_train[var_set].mean()
    df_train[var_set] = df_train[var_set].fillna(train_means)
    df_test[var_set] = df_test[var_set].fillna(train_means)

    # Impute BMI as mean within age and gender group
    print(f"No. BMI values imputed by mean (train): {df_train.bmi.isna().sum()}")
    df_train["age_bucket"], bin_edges = pd.qcut(df_train.alder, 3, ["low", "medium", "high"], retbins=True)
    mean_bmi_train = df_train.groupby(["sex", "age_bucket"], observed=True)["bmi"].mean()
    df_train = df_train.set_index(["sex", "age_bucket"])
    df_train["bmi"] = df_train["bmi"].fillna(mean_bmi_train)
    df_train = df_train.reset_index()

    # Repeat for test set
    df_test["age_bucket"] = pd.cut(df_test.alder, bins=bin_edges, labels=["low", "medium", "high"], include_lowest=True)
    df_test = df_test.set_index(["sex", "age_bucket"])
    df_test["bmi"] = df_test["bmi"].fillna(mean_bmi_train)
    df_test = df_test.reset_index()

    return df_train, df_test


def train_test_impute_tv(df_event_tv, var_set, gr=1):
    var_set_temp = var_set + ["LopNr", "bmi"]
    dict_sets = {}

    # Apply same split and imputation as in time-independent setting
    for set in ["train", "test"]:
        if gr==1:
            df_set = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times_{set}.csv")
        elif gr==2:
            df_set = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times_{set}.csv")

        df_event_tv_set = df_event_tv[df_event_tv.LopNr.isin(df_set.LopNr.unique())].sort_values(["LopNr", "start"])

        diag_rows_set = df_event_tv_set.groupby("LopNr").first().reset_index()[var_set_temp + ["start"]].merge(df_set[var_set_temp], on="LopNr", how="inner", suffixes=("_old", "_new")).set_index(["LopNr", "start"])
    
        df_event_tv_set.set_index(["LopNr", "start"], inplace=True)
        for var in var_set_temp:
            if var != "LopNr":
                df_event_tv_set[var] = df_event_tv_set[var].where(df_event_tv_set[var].notna(), diag_rows_set[f"{var}_new"])

        df_event_tv_set.reset_index(inplace=True)

        # For now, forward fill the rest of the rows
        df_event_tv_set = df_event_tv_set.set_index("LopNr").groupby("LopNr").ffill().reset_index()
        dict_sets[set] = df_event_tv_set

    return dict_sets


def categorize_medical_cutoffs(df_survival_times):
    # HbA1c cut-off >= 48 considered diabetic
    df_survival_times["hba1c_high"] = df_survival_times.hba1c >= 48
    df_survival_times["hba1c_low"] = df_survival_times.hba1c < 48

    # GFR < 60 means kidney disease
    df_survival_times["GFR_high"] = df_survival_times.GFR > 60
    df_survival_times["GFR_low"] = df_survival_times.GFR <= 60

    # Cholesterol < 5.17 considered normal
    df_survival_times["kolesterol_high"] = df_survival_times.kolesterol >= 5.17
    df_survival_times["kolesterol_low"] = df_survival_times.kolesterol < 5.17
    
    # Systolic blood pressure >+ 130 high
    df_survival_times["systoliskt_high"] = df_survival_times.systoliskt >= 130
    df_survival_times["systoliskt_low"] = df_survival_times.systoliskt < 130

    return df_survival_times


#%% 
# Define variables
var_set = ["hba1c", "GFR", "rokare", "kolesterol", "systoliskt"] # Dropped HDL for >30% missingness
soc_ec_var = ["Civil", "HushallsTyp_RTB", "Sun2000niva_old", "DispInkKEHB04", "Raks_AndelArblosInk", "Raks_AndelSjukInk", "Raks_AndelEkBisInk"]
drug_var = ["antithrombotic_agents", "antihypertensive_comb", "lipid_modifying_agents"]
sleep_var = ["IV_AHI", "IV_ODI", "IV_AverageSaturation"]
full_var_set = var_set + soc_ec_var + drug_var + ["FodelseLand_EU27_2020"]

#%%
# Create subsets
# Drop unused columns
df_t2d_diag = df_t2d_diag[["LopNr", "regdat", "sex", "alder", "vikt", "langd", "bmi"] + var_set]

# Merge with socio-economic data
df_t2d_diag_w_soc_ec = merge_socio_economic_data(df_t2d_diag, soc_ec_var)

# Impute baseline missingness by looking at future measurements
df_t2d_diag_w_soc_ec_time_imp = impute_within_time_range(df_t2d_diag_w_soc_ec, df_t2d, var_set)

# Fill in drug usage
df_t2d_diag_w_drugs = compute_drug_usage(df_t2d_diag_w_soc_ec_time_imp)
df_t2d_diag_w_drugs.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/t2d_diag_w_drugs.csv")

# Get survival analysis format
df_survival_times = to_surv_analysis_format(df_t2d_diag_w_drugs, plot_missingness=False, var_set=full_var_set)

# Filter out NDR+SESAR subset and merge some SESAR data
df_survival_times_sub = df_survival_times[df_survival_times.LopNr.isin(df_osa_diag.LopNr)]
df_survival_times_sub = df_survival_times_sub.merge(df_osa_diag[["LopNr"] + sleep_var], on="LopNr", how="inner")

# Compute categorized medical cut-off variables
df_survival_times = categorize_medical_cutoffs(df_survival_times)
df_survival_times_sub = categorize_medical_cutoffs(df_survival_times_sub)

# Save to file
df_survival_times.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times.csv", index=False)
df_survival_times_sub.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times.csv", index=False)

#%% 
# Train/test split
df_survival_times = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times.csv")
df_survival_times_sub = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times.csv")

# Train test split and save to file
df_train, df_test = train_test_impute(df_survival_times, full_var_set, train_size=0.75)
df_train_sub, df_test_sub = train_test_impute(df_survival_times_sub, full_var_set + sleep_var, train_size=0.75)

df_train = categorize_medical_cutoffs(df_train)
df_test = categorize_medical_cutoffs(df_test)
df_train_sub = categorize_medical_cutoffs(df_train_sub)
df_test_sub = categorize_medical_cutoffs(df_test_sub)

df_train.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times_train.csv", index=False)
df_test.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times_test.csv", index=False)

df_train_sub.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times_train.csv", index=False)
df_test_sub.to_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times_test.csv", index=False)

#%% 
# Generate time-varying data
df_survival_times = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times.csv")
df_survival_times_sub = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times.csv")

# Process longitudinal data
df_t2d = df_t2d.sort_values(["LopNr", "regdat"])
df_t2d["diag_date"] = df_t2d.groupby("LopNr")["regdat"].transform("min")
df_t2d_wo_diag = df_t2d[df_t2d.diag_date != df_t2d.regdat].sort_values(["LopNr", "regdat"])
df_t2d_wo_diag["time_since_diag"] = (df_t2d_wo_diag.regdat - df_t2d_wo_diag.diag_date).dt.days
df_t2d_wo_diag.loc[df_t2d_wo_diag.bmi.isnull() & ((df_t2d_wo_diag.vikt.notnull() & df_t2d_wo_diag.langd.notnull())), "bmi"] = df_t2d_wo_diag.loc[df_t2d_wo_diag.bmi.isnull() & ((df_t2d_wo_diag.vikt.notnull() & df_t2d_wo_diag.langd.notnull())), "vikt"] / (df_t2d_wo_diag.loc[df_t2d_wo_diag.bmi.isnull() & ((df_t2d_wo_diag.vikt.notnull() & df_t2d_wo_diag.langd.notnull())), "langd"] / 100) ** 2

# Get time-varying data and save
dict_event_tv = to_time_varying(df_survival_times, df_t2d_wo_diag, var_set)
dict_event_tv_sub = to_time_varying(df_survival_times_sub, df_t2d_wo_diag, var_set)

for event in ["death", "ami", "stroke"]:
    dict_event_tv[event].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv.csv", index=False)
    dict_event_tv_sub[event].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv.csv", index=False)

#%%
# Train-test split and impute in time-varying data
for event in ["death", "ami", "stroke"]:
    print(f"Processing {event}...")
    df_event_tv = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv.csv")
    df_event_sub_tv = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv.csv")

    dict_sets = train_test_impute_tv(df_event_tv, full_var_set)
    dict_sub_sets = train_test_impute_tv(df_event_sub_tv, full_var_set + sleep_var, gr=2)

    dict_sets["train"] = categorize_medical_cutoffs(dict_sets["train"])
    dict_sets["test"] = categorize_medical_cutoffs(dict_sets["test"])

    dict_sub_sets["train"] = categorize_medical_cutoffs(dict_sub_sets["train"])
    dict_sub_sets["test"] = categorize_medical_cutoffs(dict_sub_sets["test"])
    
    dict_sets["train"].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_train.csv", index=False)
    dict_sets["test"].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_test.csv", index=False)

    dict_sub_sets["train"].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv_train.csv", index=False)
    dict_sub_sets["test"].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv_test.csv", index=False)

    print(f"{event} done\n")

# %%
# To wide format for testing
for event in ["death", "ami", "stroke"]:
    print(f"Processing {event}...")

    df_train = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_train.csv")
    df_test = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_test.csv")

    df_sub_train = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv_train.csv")
    df_sub_test = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv_test.csv")

#%% 
# Remove tv variable one at a time for testing
def tv_one_out(df_event_tv, var_set, gr=1):
    for var in var_set:
        print(f"\tProcessing {var}...")

        # Replace all var values with baseline value, and reset categorical variables 
        baseline_val = df_event_tv.groupby("LopNr").first()[var]
        df_event_tv = df_event_tv.drop(columns=[var])
        df_event_tv = df_event_tv.merge(baseline_val, on="LopNr", how="left")
        df_event_tv = categorize_medical_cutoffs(df_event_tv)

        # Number of rows with duplications
        identical_cols = df_event_tv.columns.difference(["start", "stop", "event"])

        # Create a key per identical row
        df_event_tv["_cov_key"] = df_event_tv[identical_cols].apply(lambda row: hash(tuple(row)), axis=1)
        
        # Create a grouping for each continuous interval of identical rows
        df_shifted = df_event_tv.shift()
        new_group = (
            (df_event_tv['_cov_key'] != df_shifted['_cov_key']) |
            (df_event_tv['start'] != df_shifted['stop']) |
            (df_event_tv['LopNr'] != df_shifted['LopNr'])  # ensure matching id
        )
        df_event_tv["_group"] = new_group.cumsum()

        # Merge each group into one row
        df_event_tv = (
            df_event_tv.groupby("_group")
            .agg({"LopNr": "first", 
                    "start": "min", 
                    "stop": "max", 
                    "event": "last", 
                    **{col: "first"for col in identical_cols}})
                    .reset_index(drop=True)
                    )
        
        ordered_cols = ["LopNr", "start", "stop", "event"] + [col for col in df_event_tv.columns if col not in ["LopNr", "start", "stop", "event"]]
        print(("_cov_key" in ordered_cols) | ("_group" in ordered_cols))
        if gr == 1:
            df_event_tv[ordered_cols].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_min_{var}_train.csv")
        else:
            df_event_tv[ordered_cols].to_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_min_{var}_sub_train.csv")

            
for event in ["death", "ami", "stroke"]:
    print(f"Processing {event}...")

    # Load event data
    df_train = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_tv_train.csv").sort_values(["LopNr", "start"])
    df_sub_train = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/{event}_sub_tv_train.csv").sort_values(["LopNr", "start"])

    tv_one_out(df_train, var_set + ["bmi"])
    tv_one_out(df_sub_train, var_set + ["bmi"], gr=2)
 

# %%
