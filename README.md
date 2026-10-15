# Arctic Ice Analysis
This project explores the relationship between Arctic sea ice and temperature using public ice data from NSIDC and temperature data from NASA GISTEMP. 

# Research Question
*How is the loss of Arctic sea ice correlated with Arctic temperature?*

# Data Sources
Sea Ice:
- File: `N_seaice_extent_daily_v4.0.csv`
- Source: Daily Arctic sea ice extent (NSIDC Sea Ice Index, G02135 v4): <https://nsidc.org/data/g02135/versions/4
- Citation: Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., Windnagel, A. K. & Stafford, T. (2025). Sea Ice Index. (G02135, Version 4). [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. DOI: 10.7265/a98x-0f50.

Temperature: 
- File: `ZonAnn.Ts+dSST.csv`
- Source: Zonal annual mean temperature anomalies, °C relative to 1951–1980 (NASA GISTEMP v4): <https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv
- Citation: Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble. J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179, DOI:10.1029/2023JD040179. 

# What the script does
- Cleans and prepares the ice and temperature data
- Combines data for each year:
    - Average September sea ice from that year's daily September measurements
    - Average yearly temperature for Arctic lattitudes (64N-90N)
- Visualises:
    - September sea ice for every year with uncertainty and trendline
    - Arctic temperature for every year with trendline
    - September sea ice against temperature with Pearson correlation coefficient

# Output
The results will be printed in the terminal. A printed warning like `skipping 1 rows with missing/invalid data` is expected. It is the row of the ice data containing the units. 

A folder titled `results - ice v temp` will be created in the same location as the script. The following figures and cleaned data will be saved there. 

- `cleaned_analysed_data.csv`
    Joined ice and temperature data used in the analysis. Columns for year, September ice mean and daily std, and temperature. 
- `fig1_ice_and_temperature.png`
    Annual September sea ice and temperature both plotted against year with their own fitted trends.
    * The ice trendline is a linear fit for the September ice amount versus year, in million km² per decade.
    * The temperature trendline is a linear fit temperature difference compared to 1951-80 versus year in °C per decade.
    * Solid shading around the trendlines shows statistical spread of subsample slopes. 
    * Hashed shading shows the systematic range sensitivity of each trend slope.
    * The vertical bars on the ice datapoints show the std of each year's average September ice extent. 
- `fig2_correlation.png`
    Temperature versus September sea ice. 
    * The vertical bars show the std of daily September ice measurements for each year. 
    * The trendline is the Pearson correlation coefficient (r). 

# How to run
1. Make sure all needed packages are installed:
```bash
pip install numpy pandas matplotlib scipy
```
2. Download data files from sources above 
3. Put the script, the ice data, and the temperature data in the same folder in VS Code. Ensure names match:
    - `ice_temp_analysis.py`
    - `N_seaice_extent_daily_v4.0.csv`
    - `ZonAnn.Ts+dSST.csv`
4. Run the script
```bash
python3 ice_temp_analysis.py
```

# Customization (optional)
At the top of the code, there are global variables that can be easily changed. This includes the number of subsamples (`N_SUBSAMPLES`), output file names, and time range. 

Setting `SELF_TESTS=True` will run only the self tests to check the subsampling and slope functions. The rest of the script will not run. 
