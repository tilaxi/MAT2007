"""
MAT2007 PROJECT -- Arctic sea ice vs temperature

QUESTION: How is the loss of Arctic sea ice correlated with Arctic temperature?

DATA
  - Sea Ice
        Name:   NSIDC Sea Ice Index v4, daily Arctic extent
        File:   N_seaice_extent_daily_v4.0.csv
        Cite:   Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., Windnagel, A. K. & Stafford, T. (2025). Sea Ice 
                  Index. (G02135, Version 4). [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. 
                  DOI: 10.7265/a98x-0f50. Accessed 2026-09-27 at https://nsidc.org/data/g02135/versions/4 
  - Temperature
        Name:   NASA GISTEMP v4, zonal annual means anomaly in deg C relative to 1951-1980
        File:   ZonAnn.Ts+dSST.csv  
        Cite:   GISTEMP Team, 2026: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA Goddard 
                  Institute for Space Studies. Dataset accessed 2026-09-27 at https://data.giss.nasa.gov/gistemp/.
                Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: A GISTEMPv4 
                  observational uncertainty ensemble. J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179, DOI:10.1029/2023JD040179. 

INSTRUCTIONS
- Put this file and the two csv files in one folder together and run the code.
- The code will create and save the outputs to a new folder called "results - ice v temp" at the current code location. 
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# ################### #
#        SETUP        #
# ################### #
SCRIPT_FOLDER = Path(__file__).resolve().parent                   # the folder this file is in
ICE_FILE = SCRIPT_FOLDER / "N_seaice_extent_daily_v4.0.csv"       # NSIDC data for daily sea ice extent (source file)
TEMP_FILE = SCRIPT_FOLDER / "ZonAnn.Ts+dSST.csv"                  # GISTEMP data for zonal annual temperature means (source file)
OUTPUT_FOLDER = SCRIPT_FOLDER / "results - ice v temp"            # figures saved here

SELF_TESTS = False      # runs ONLY the self tests when True

N_SUBSAMPLES = 10       # number of piles to be used for the subsampling method
SEED = 69               # (for consistency)
MIN_DAYS = 10           # a given September needs at least this many days of data to be included
START_YEAR = 1979       # 1979 is the first full year of satellite ice data available
SLOPE_RANGES = (        # ranges used to calculate systematic uncertainty of the slope. Ranges must be > 2 years. 
    (1979, 2025), 
    (1979, 2015), 
    (1989, 2025))          



# ################### #
#    DATA CLEANING    #
# ################### #

def load_september_ice(path):
    # One row / year for the mean September extent and its statistical uncertainty (in million km^2).
    # The statistical uncertainty of a year = the std of the daily values that went into the september mean.
    
    df = pd.read_csv(path, skipinitialspace=True)      # reads the file as a table
    df.columns = [c.strip() for c in df.columns]       # removes stray spaces in column names
    
    required = {"Year", "Month", "Day", "Extent"}      # makes sure the file has the required columns
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sea ice CSV file is missing required columns: {sorted(missing)}")
    
    numeric_df = df[["Year", "Month", "Day", "Extent"]].apply(pd.to_numeric, errors="coerce")

    finite_rows = np.isfinite(numeric_df).all(axis=1)
    invalid_rows = numeric_df.isna().any(axis=1) | ~finite_rows

    if invalid_rows.any():                              # report any missing rows/years.
        warnings.warn(
            f"Skipping {invalid_rows.sum()} rows with missing/invalid data",
            stacklevel=2)

    numeric_df = numeric_df.loc[~invalid_rows]
    sept = numeric_df[numeric_df["Month"] == 9]                         # keeps only the September rows
    g = sept.groupby("Year")["Extent"].agg(["mean", "std", "count"])   # average, std, and number of days for September in each year
    return g[g["count"] >= MIN_DAYS][["mean", "std"]]   # drops years with too few days


def load_arctic_temp(path):
    # Arctic (64N-90N) temperature per year relative to 1951-80, in deg C.
    # cleans data to exclude missing/infinite values, and invalid/duplicate years
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    required = {"Year", "64N-90N"}                      # makes sure the file has the required columns
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Temp CSV file is missing required columns: {sorted(missing)}")
        
    temp = pd.to_numeric(df["64N-90N"], errors="coerce")            # any missing temps become NaN
    year = pd.to_numeric(df["Year"], errors="coerce")               # any missing years become NaN
    
    numeric_df = pd.DataFrame({"year": year, "temp": temp})

    invalid_rows = numeric_df.isna().any(axis=1)                    # reports any missing rows/years. 
    if invalid_rows.any():
        warnings.warn(
            f"Skipping {invalid_rows.sum()} rows with missing/invalid data",
            stacklevel=2)

    clean_temp = numeric_df.loc[~invalid_rows].copy()

    if not np.isfinite(clean_temp[["year", "temp"]].to_numpy()).all():   
        raise ValueError("Temp data contains non-finite values.")

    if not np.isclose(clean_temp["year"], clean_temp["year"].round()).all():  
        raise ValueError("Temp data contains a year that is not an integer")

    clean_temp["year"] = clean_temp["year"].astype(int)                      

    if clean_temp["year"].duplicated().any():
        raise ValueError("Temp data has duplicate years.")

    return clean_temp.set_index("year")["temp"].sort_index()


# ################## #
#     STATISTICS     #
# ################## #
#   statistical uncertainty = std of the subsample means (or plain std for a small sample)
#   error = statistical uncertainty / sqrt(N)   (not used in this script)

def subsampling(values, stat_func=np.mean, n_sub=N_SUBSAMPLES, seed=SEED):
    # shuffles, splits into piles, finds statistical uncertainty
    # returns value, and statistical uncertainty

    vals = np.asarray(values, dtype=float)

    # ensures input is big enough so that extra empty piles aren't created. 
    if vals.ndim == 0 or vals.size == 0:
        raise ValueError("There are no values to subsample.")
    if not np.isfinite(vals).all():
        raise ValueError("values contain infinite numbers")
    if not isinstance(n_sub, int) or n_sub < 2:
        raise ValueError("n_sub is not an integer >= 2")
    if n_sub > vals.size:
        raise ValueError("n_sub is larger than the number of observations")

    rng = np.random.default_rng(seed)                      # random generator with a fixed seed
    piles = np.array_split(rng.permutation(vals), n_sub)   # shuffles, then splits into n_sub piles
    results = np.array([stat_func(p) for p in piles])      # the mean of each pile
    stat_unc = np.std(results, ddof=1)                     # statistical uncertainty
    return float(stat_func(vals)), float(stat_unc)

def slope_per_decade(pairs):
    #straight line slope of a quantity vs year, per decade (ice units: million km², temperature units: °C)
    pairs = np.asarray(pairs, dtype=float)    # a table with one row per year: (year, ice extent)
    return float(10 * np.polyfit(pairs[:, 0], pairs[:, 1], 1)[0])


def slope_systematic_uncertainty(years, values):
    # fits the slope over 3 different year ranges
    # statistic uncertainty of slope = 1/2 the spread of the three slopes
    slopes = []

    for first, last in SLOPE_RANGES:
        keep = (years >= first) & (years <= last)        # only the years in this range
        if keep.sum() < 2:
            raise ValueError(f"The year range {first} - {last} is not large enough to provide an accurate uncertainty.")
        slopes.append(slope_per_decade(np.column_stack([years[keep], values[keep]])))
    
    return (max(slopes) - min(slopes)) / 2               # half the spread of the three results


# ################### #
#        PLOTS        #
# ################### #
def trend_line(years, values, slope_dec, unc_dec):
    # adds a line through the middle of the data with the fitted slope, 
    # plus the lines for slope +/- its statistical uncertainty (the edges of the shaded band)
    x0, y0 = years.mean(), values.mean()
    line = y0 + slope_dec / 10 * (years - x0)
    
    low = y0 + (slope_dec - unc_dec) / 10 * (years - x0)
    high = y0 + (slope_dec + unc_dec) / 10 * (years - x0)
    
    return line, low, high

def sigma_txt(slope, uncertainty):          # calculation that prevents division by zero error
    if not np.isfinite(uncertainty) or uncertainty <=0:
        return "not available"
    return f"{abs(slope) / uncertainty:.1f} sigma"

# fig 1: x = year, y1 = ice extent, y2 = temperature
def plot_timeseries(years, ice, ice_unc, temp, ice_trend, temp_trend, name):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.errorbar(years, ice, yerr=ice_unc, fmt="o", ms=3, 
                color="tab:blue", ecolor="tab:blue", alpha=0.8, capsize=2, 
                label="September ice (± std of the days)")
    
    line, low, high = trend_line(years, ice, * ice_trend[:2])
    
    ax.plot(years, line, color="tab:blue", lw=2,
            label=f"Ice trend {ice_trend[0]:+.2f} ± {ice_trend[1]:.2f} (stat) ± {ice_trend[2]:.2f} (syst) per decade, {sigma_txt(ice_trend[0], ice_trend[1])}")
    
    ax.fill_between(years, low, high, color="tab:blue", alpha=0.2)      # band = statistical uncertainty of the slope
    _, ice_sys_low, ice_sys_high = trend_line(years, ice, ice_trend[0], ice_trend[2])
    ax.fill_between(years, ice_sys_low, ice_sys_high,
                    facecolor="none", edgecolor="tab:blue", hatch="///",
                    label="Ice slope range sensitivity")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("September Ice Extent (million km²)", color="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(years, temp, "o", ms=3, color="tab:orange", alpha=0.8, 
             label="Relative Arctic temperature")
    line, low, high = trend_line(years, temp, *temp_trend[:2])
    
    ax2.plot(years, line, color="tab:orange", lw=2,
             label=f"Temperature trend {temp_trend[0]:+.2f} ± {temp_trend[1]:.2f} (stat) ± {temp_trend[2]:.2f} (syst) °C per decade, {sigma_txt(temp_trend[0], temp_trend[1])}")
    
    ax2.fill_between(years, low, high, color="tab:orange", alpha=0.2)
    _, temp_sys_low, temp_sys_high = trend_line(years, temp, temp_trend[0], temp_trend[2])
    ax2.fill_between(years, temp_sys_low, temp_sys_high,
                     facecolor="none", edgecolor="tab:orange", hatch="xxx",
                     label="Temperature slope range sensitivity")
    
    ax2.set_ylabel("Relative Arctic Temperature (°C)", color="tab:orange")
    
    ax.set_title(f"Arctic September Sea Ice and Temperature vs. Year "
                f"({int(years.min())}-{int(years.max())})")
    ax.grid(alpha=0.2)
    
    # one legend for both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)   
    
    # note at bottom
    fig.text(0.01, 0.005,
        "Solid shading = statistical slope spread\n\
        Hashing = systematic slope range sensitivity",
        fontsize=7, color="grey")
    
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(name, format="png", dpi=300, bbox_inches="tight") 
    plt.close(fig)


# fig 2: x = temperature, y = ice 
def plot_correlation(temp, ice, ice_unc, r, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.errorbar(temp, ice, yerr=ice_unc, fmt="o", ms=4, color="tab:blue", ecolor="tab:blue",
                alpha=0.8, capsize=2)                  # vertical bars = std of the September days
    m, b = np.polyfit(temp, ice, 1)                    # straight line through the points (visual guide only)
    xs = np.linspace(temp.min(), temp.max(), 100)
    
    ax.plot(xs, m * xs + b, color="black", lw=1.5)
    
    ax.set_xlabel("Relative Arctic Temperature (°C)")
    ax.set_ylabel("September Ice Extent (million km²)")
    ax.set_title(f"Temperature vs September Ice Extent\nr = {r:.2f}")
    
    ax.grid(alpha=0.2)

    # note at bottom
    fig.text(0.01, 0.01, "Bars = std of daily September values.\nNote: no uncertainty for single temperature years.",
             fontsize=7, color="grey")
    
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(name, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ################### #
#       TESTING       #
# ################### #
def self_tests():
    m, e = subsampling(np.arange(40.0))
    assert abs(m - 19.5) < 1e-12 and e > 0
    
    x = np.arange(20.0)
    assert abs(                                 # slope 2 per year = 20 per decade
        slope_per_decade(np.column_stack([x, 2 * x + 1])) - 20.0) < 1e-9   
    
    m, e = subsampling(
        np.column_stack([x, 2 * x + 1]), 
        stat_func=slope_per_decade, 
        n_sub=4)
    assert abs(m - 20.0) < 1e-9 and e < 1e-9     # a perfect line gives the same slope in every pile
    
    print("Self tests: PASSED\n")                # will fail if asserts are not true


# #################### #
#         MAIN         #
# #################### #

def main():
    if SELF_TESTS:                                      # run self tests if chosen at top of code
        self_tests()
        return

    OUTPUT_FOLDER.mkdir(exist_ok=True)                   # make the results folder if it doesn't exist yet
    
    ice_table = load_september_ice(ICE_FILE)             # columns are mean and std per year
    ice = ice_table["mean"]
    temp = load_arctic_temp(TEMP_FILE)
    
    both = pd.concat(       # puts ice and temperature side by side, and keeps only years that have data for both
        [ice.rename("ice"), ice_table["std"].rename("ice_unc"), temp.rename("temp")],
         axis=1, join="inner").dropna()
    both = both[both.index >= START_YEAR]

    if len(both) < N_SUBSAMPLES:                        # makes sure there are enough years with valid data
        raise ValueError(
            f"There are only {len(both)} paired years of data.\n\
            At least {N_SUBSAMPLES} are needed for subsampling.")

    cleaned_data_file = OUTPUT_FOLDER / "cleaned_analysed_data.csv"
    both.to_csv(cleaned_data_file, index_label="Year")
    print(f"Saved cleaned data used in analysis to: {cleaned_data_file}")

    years = both.index.values
    ice_v, temp_v, ice_unc = both["ice"].values, both["temp"].values, both["ice_unc"].values
    print(f"\nAnalysing {len(years)} paired years: {int(years.min())}-{int(years.max())}\n")

    # ice loss trend
    pairs = np.column_stack([years, ice_v])                                # one row per year: (year, ice)
    ice_slope, ice_stat = subsampling(pairs, stat_func=slope_per_decade)   # slope of each "pile" of years
    ice_syst = slope_systematic_uncertainty(years, ice_v)
    print(f"Ice loss since {int(years.min())}: \n\
            {ice_slope:+.2f} ± {ice_stat:.2f} (stat unc),"
            f"± {ice_syst:.2f} (syst unc) million km² per decade, "
            f"({sigma_txt(ice_slope, ice_stat)} from zero)")            # sigma = value / statistical uncertainty

    # temperature trend 
    t_pairs = np.column_stack([years, temp_v])
    t_slope, t_stat = subsampling(t_pairs, stat_func=slope_per_decade)
    t_syst = slope_systematic_uncertainty(years, temp_v)
    print(f"\nTemperature change since {int(years.min())}:\n\
            {t_slope:+.2f} ± {t_stat:.2f} (stat unc) "
            f"± {t_syst:.2f} (syst unc) °C per decade, "
            f"({sigma_txt(t_slope, t_stat)} from zero)")

    # comparing ice vs temp
    r_raw = stats.pearsonr(temp_v, ice_v)[0]                            # r near -1 = warmer years match up with less ice (not proof of cause)
    print(f"\nCorrelation of temperature vs ice: r = {r_raw:.2f}\n")

    # plotting
    plot_timeseries(years, ice_v, ice_unc, temp_v, (ice_slope, ice_stat, ice_syst),
                    (t_slope, t_stat, t_syst), OUTPUT_FOLDER / "fig1_ice_and_temperature.png")
    plot_correlation(temp_v, ice_v, ice_unc, r_raw, OUTPUT_FOLDER / "fig2_correlation.png")


if __name__ == "__main__":
    main()
