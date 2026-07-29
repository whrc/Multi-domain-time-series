# Data Dictionary: RangeSTAR Ecosystem Carbon Flux Predictions and Observations

This repository contains observational data and ecosystem-scale predictions of carbon fluxes and pools across several rangeland sites within the AmeriFlux and National Ecological Observatory Network (NEON) registries. 

The dataset combines satellite remote sensing inputs, gridded meteorological drivers (NLDAS, Daymet), field tower-based eddy covariance observations, and model-derived carbon cycle predictions (including uncertainty intervals and individual respiratory/biomass pools).

## 1. Data Dictionary

### Section A: Core Identifiers & Metadata
| Column Name | Data Type | Units | Description |
| :--- | :---: | :---: | :--- |
| `site` | String | *N/A* | Unique identifier code matching the AmeriFlux / NEON site registry. |
| `time` | Date | YYYY-MM-DD | The date of the daily integrated observation/prediction. |
| `Site label_x` | String | *N/A* | Source reference label for the site used during dataset merges. |
| `Site label_y` | String | *N/A* | Source reference label for the site used during dataset merges. |
| `PFT` | String | *N/A* | Plant Functional Type classification. |

### Section B: Remote Sensing & Soil Characteristics
| Column Name | Data Type | Units | Description | Source |
| :--- | :---: | :---: | :--- |  :--- |
| `EVI2` | Float | Dimensionless | **Enhanced Vegetation Index 2** derived from satellite imagery. Indicates canopy greenness/density. Range: `-0.10` to `0.38`. | Landsat-MODIS STARFM
| `tsoil` | Float | °C | **Soil Temperature** measured near the surface. Range: `-12.60°C` to `36.01°C`. | NLDAS
| `sm1` | Float | m³ H₂O / m³ soil | **Volumetric Soil Moisture Layer 1** (shallow surface layer water content). Range: `0.08` to `0.44`. | NLDAS
| `sm2` | Float | m³ H₂O / m³ soil | **Volumetric Soil Moisture Layer 2** (deeper sub-surface layer water content/index). Range: `0.37` to `1.32`. | NLDAS 
| `clay` | Float | % | **Clay Content** percentage of the localized soil profile. Range: `6.63%` to `29.30%`. | SOLUS

### Section C: Gridded Meteorological Drivers
| Column Name | Data Type | Units | Description | Source
| :--- | :---: | :---: | :--- |  :--- |
| `vpd` | Float | hPa | **Vapor Pressure Deficit** from gridded reanalysis data (see Note 2 re: unit mismatch with `VPD_obs`). | DAYMET 
| `SW_IN_NLDAS` | Float | Variable / Scaled | **Incoming Downward Shortwave Radiation** from the NLDAS model grid. Highly correlated with measured solar flux. | NLDAS
| `tavg` | Float | °C | **Daily Average Air Temperature** derived from gridded atmospheric data. | Daymet
| `tmax` | Float | °C | **Daily Maximum Air Temperature** derived from gridded atmospheric data. | Daymet
| `tmin` | Float | °C | **Daily Minimum Air Temperature** derived from gridded atmospheric data. | Daymet
| `prcp` | Float | mm / day | **Daily Total Precipitation** derived from gridded atmospheric data. | Daymet

### Section D: In-Situ Observed Fluxes & Weather (Eddy Covariance Tower)
*Note: Observed flux variables may contain missing values (`NaN`) due to temporary instrument down-time or quality filtering.*

| Column Name | Data Type | Units | Description |
| :--- | :---: | :---: | :--- |
| `NEE_obs` | Float | g C m⁻² d⁻¹ | **Observed Net Ecosystem Exchange**. Net CO₂ flux between the ecosystem and atmosphere (sign convention: see Note 1). |
| `GPP_obs` | Float | g C m⁻² d⁻¹ | **Observed Gross Primary Productivity**. Total photosynthetic carbon capture by the canopy. Always $\ge 0$. |
| `RECO_obs` | Float | g C m⁻² d⁻¹ | **Observed Ecosystem Respiration**. Total biotic carbon release (autotrophic + heterotrophic). Always $\ge 0$. |
| `TA_obs` | Float | °C | **Observed Air Temperature** measured directly at the tower canopy level. |
| `SW_IN_obs` | Float | W m⁻² | **Observed Incoming Shortwave Radiation** (solar flux) at the tower radiometer. |
| `P_obs` | Float | mm / day | **Observed Daily Precipitation** total recorded by the tower rain gauge. |
| `PA_obs` | Float | kPa | **Observed Atmospheric Pressure** at the tower site. |
| `RH_obs` | Float | % | **Observed Relative Humidity** of the air column. |
| `VPD_obs` | Float | kPa | **Observed Vapor Pressure Deficit** computed directly from tower instruments (see Note 2 re: unit mismatch with `vpd`). |
| `NETRAD_obs` | Float | W m⁻² | **Observed Net Radiation** balancing incoming/outgoing shortwave and longwave radiation. |
| `WS_obs` | Float | m s⁻¹ | **Observed Wind Speed** measured by the tower anemometer. |
| `H_obs` | Float | W m⁻² | **Observed Sensible Heat Flux**. Heat energy transferred from the surface to the air via conduction/convection. |
| `LE_obs` | Float | W m⁻² | **Observed Latent Heat Flux**. Energy consumed by evapotranspiration from the surface. |

### Section E: Model-Predicted Carbon Fluxes & Uncertainty Intervals
| Column Name | Data Type | Units | Description |
| :--- | :---: | :---: | :--- |
| `NEE_predicted` | Float | g C m⁻² d⁻¹ | **Predicted Net Ecosystem Exchange** from the core optimized predictive model. |
| `NEE_original` | Float | g C m⁻² d⁻¹ | **Original Baseline NEE** prediction before calibration or ensemble processing. |
| `NEE_pred_mean` | Float | g C m⁻² d⁻¹ | **Ensemble Mean Predicted NEE** representing the expected value across the full uncertainty run. |
| `NEE_pred_lower_95` | Float | g C m⁻² d⁻¹ | Lower bound of the **95% Confidence / Prediction Interval** for NEE. |
| `NEE_pred_upper_95` | Float | g C m⁻² d⁻¹ | Upper bound of the **95% Confidence / Prediction Interval** for NEE. |
| `NEE_pred_lower_68` | Float | g C m⁻² d⁻¹ | Lower bound of the **68% Confidence / Prediction Interval** (approx. $\pm 1 \sigma$) for NEE. |
| `NEE_pred_upper_68` | Float | g C m⁻² d⁻¹ | Upper bound of the **68% Confidence / Prediction Interval** (approx. $\pm 1 \sigma$) for NEE. |
| `GPP_predicted` | Float | g C m⁻² d⁻¹ | **Predicted Gross Primary Productivity** representing total canopy photosynthesis. |
| `RECO_predicted` | Float | g C m⁻² d⁻¹ | **Predicted Ecosystem Respiration** ($R_{eco} = R_m + R_g + \text{heterotrophic component}$). |
| `Rm_predicted` | Float | g C m⁻² d⁻¹ | **Predicted Maintenance Respiration** of the plant community. |
| `Rg_predicted` | Float | g C m⁻² d⁻¹ | **Predicted Growth Respiration** associated with new tissue development. |

### Section F: Model-Predicted Biomass & Carbon Pools
| Column Name | Data Type | Units | Description |
| :--- | :---: | :---: | :--- |
| `AGB_predicted` | Float | g C m⁻² | **Predicted Aboveground Biomass** carbon pool (stems, leaves, structural tissue). |
| `BGB_predicted` | Float | g C m⁻² | **Predicted Belowground Biomass** carbon pool (roots, structural belowground tissue). |
| `AGL_predicted` | Float | g C m⁻² | **Predicted Aboveground Litter** carbon pool (fallen leaves, dead surface organic material). |
| `BGL_predicted` | Float | g C m⁻² | **Predicted Belowground Litter** carbon pool (dead roots, fine organic matter below surface). |
| `POC_predicted` | Float | g C m⁻² | **Predicted Particulate Organic Carbon** pool in the soil matrix (fast-cycling). |
| `HOC_predicted` | Float | g C m⁻² | **Predicted Humus Organic Carbon** pool in the soil matrix (slow-cycling, passive/protected carbon). |

---

## 2. Key Methodological & Technical Notes

1. **Carbon Flux Convention:** - Following standard micro-meteorological conventions, **Net Ecosystem Exchange (NEE)** views the atmosphere as the reference frame. 
   - A **negative value** (e.g., `-2.5 g C m⁻² d⁻¹`) means a net drawdown of CO₂ from the atmosphere into the ecosystem (a carbon **sink**). 
   - A **positive value** indicates a net release of carbon into the atmosphere (a carbon **source**).
   - **GPP** and **RECO** are universally defined as positive rates where $\text{NEE} = \text{RECO} - \text{GPP}$.

2. **Vapor Pressure Deficit (VPD) Discrepancy:**
   - Note that `vpd` (gridded driver) is recorded in **hPa** (values scale around 0–40), while `VPD_obs` (tower tower observation) is recorded in **kPa** (values scale around 0–4). Ensure appropriate conversion ($1 \text{ kPa} = 10 \text{ hPa}$) if directly comparing or calculating residuals.

3. **Pools vs. Fluxes:**
   - **Fluxes** (`NEE`, `GPP`, `RECO`, `Rm`, `Rg`) measure rates of carbon exchange over time and are expressed in **grams of Carbon per square meter per day** ($\text{g C m}^{-2} \text{d}^{-1}$).
   - **Pools** (`AGB`, `BGB`, `AGL`, `BGL`, `POC`, `HOC`) measure standing stocks of stored carbon and are expressed in **grams of Carbon per square meter** ($\text{g C m}^{-2}$).
