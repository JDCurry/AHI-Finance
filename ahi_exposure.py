"""
AHI Exposure Intelligence Module
=================================
Resilience Analytics Lab, LLC

Combines AHI precomputed monthly predictions with FEMA NRI (RAPT) expected
annual loss data to produce:
  - Conditional Expected Loss (CEL) estimates
  - Exceedance Probability (EP) curves and return-period losses
  - Deviation-from-baseline signals
  - Multi-hazard compound risk flags
  - Social vulnerability-weighted exposure
  - Month-over-month delta (June vs July)
  - Choropleth maps for all views

Usage:
    streamlit run ahi_exposure.py

Data requirements:
    data/NRI_RAPT_Counties.csv
    data/national_predictions_month06.csv
    data/national_predictions_month07.csv

CEL formula (corrected):
    Loss_given_event = NRI_EAL / NRI_Annualized_Frequency
    CEL = AHI_prob × Loss_given_event × (window_days / 365)

    Previous formula (CEL = AHI_prob × EAL × window) double-counted
    probability because EAL already encodes historical P(event).

Dollar figures sourced entirely from FEMA NRI/RAPT. Not actuarial output.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 14
NRI_PATH            = Path("data/NRI_RAPT_Counties.csv")
NFIP_SUMMARY_PATH   = Path("data/nfip_county_summary.csv")
FLOOD_TRENDS_PATH   = Path("data/flood_severity_trends.csv")
HAZARD_TRENDS_PATH  = Path("data/hazard_freq_trends.csv")
FEMA_PA_PATH        = Path("data/fema_pa_county_summary.csv")
NFIP_POLICIES_PATH  = Path("data/nfip_policies_county_summary.csv")
PREDICTIONS_DIR     = Path("data")
GEOJSON_PATH        = Path("data/national_counties.geojson")  # same file AHI uses

CREDIBILITY_K = 500  # Buhlmann credibility parameter — higher = more weight to model

AVAILABLE_MONTHS = {6: "June", 7: "July"}

HAZARD_COLS   = ['fire_p', 'flood_p', 'wind_p', 'winter_p']
HAZARD_LABELS = {'fire_p': 'Fire', 'flood_p': 'Flood', 'wind_p': 'Wind', 'winter_p': 'Winter'}
# seismic_p excluded — AHI seismic model not used in dashboard (least reliable hazard)

HIST_BASELINE = {
    'fire_p': 0.174, 'flood_p': 0.147, 'wind_p': 0.110, 'winter_p': 0.110,
}

MULTI_HAZARD_DEV_THRESHOLD = 2.0
MULTI_HAZARD_MIN_COUNT     = 2

SEVERITY_CV = 1.5  # coefficient of variation for lognormal severity assumption

NRI_COLS = [
    'State Name Abbreviation', 'County Name', 'State-County FIPS Code',
    'Population (2020)', 'Building Value ($)', 'Agriculture Value ($)',
    'Expected Annual Loss - Building Value - Composite',
    'Expected Annual Loss - Agriculture Value - Composite',
    # Per-hazard EAL
    'Wildfire - Expected Annual Loss - Building Value',
    'Wildfire - Expected Annual Loss - Agriculture Value',
    'Inland Flooding - Expected Annual Loss - Building Value',
    'Strong Wind - Expected Annual Loss - Building Value',
    'Ice Storm - Expected Annual Loss - Building Value',
    'Hurricane - Expected Annual Loss - Building Value',
    # Annualized frequency (for loss-given-event derivation)
    'Wildfire - Annualized Frequency',
    'Inland Flooding - Annualized Frequency',
    'Strong Wind - Annualized Frequency',
    'Ice Storm - Annualized Frequency',
    'Hurricane - Annualized Frequency',
    # Exposure (total value at risk per hazard)
    'Wildfire - Exposure - Building Value',
    'Inland Flooding - Exposure - Building Value',
    'Strong Wind - Exposure - Building Value',
    'Ice Storm - Exposure - Building Value',
    'Hurricane - Exposure - Building Value',
    # Historic loss ratios (for severity distribution shape)
    'Wildfire - Historic Loss Ratio - Buildings',
    'Inland Flooding - Historic Loss Ratio - Buildings',
    'Strong Wind - Historic Loss Ratio - Buildings',
    'Ice Storm - Historic Loss Ratio - Buildings',
    'Hurricane - Historic Loss Ratio - Buildings',
    # Social vulnerability and community resilience
    'Social Vulnerability - Score',
    'Social Vulnerability - Rating',
    'Community Resilience - Score',
    'Community Resilience - Rating',
]

# Mapbox tile styles (no token required — matches AHI app)
_TILE_STYLES = {
    'Dark':      {'mapbox_style': 'carto-darkmatter', 'mapbox_layers': [],
                  'border': '#30363d', 'opacity': 0.85},
    'Light':     {'mapbox_style': 'carto-positron',   'mapbox_layers': [],
                  'border': '#888',    'opacity': 0.85},
    'Satellite': {
        'mapbox_style': 'white-bg',
        'mapbox_layers': [{'below': 'traces', 'sourcetype': 'raster',
            'source': ['https://services.arcgisonline.com/ArcGIS/rest/services/'
                       'World_Imagery/MapServer/tile/{z}/{y}/{x}'],
            'sourceattribution': 'Tiles © Esri'}],
        'border': '#fbbf24', 'opacity': 0.65,
    },
}

# Color scales per metric type
_COLOR_SCALES = {
    'cel':       [[0,'#ffffcc'],[0.25,'#fed976'],[0.5,'#fd8d3c'],[0.75,'#e31a1c'],[1,'#800026']],
    'fire':      [[0,'#ffffb2'],[0.25,'#fecc5c'],[0.5,'#fd8d3c'],[0.75,'#f03b20'],[1,'#bd0026']],
    'flood':     [[0,'#f7fbff'],[0.25,'#9ecae1'],[0.5,'#4292c6'],[0.75,'#2171b5'],[1,'#084594']],
    'wind':      [[0,'#f2f0f7'],[0.25,'#bcbddc'],[0.5,'#9e9ac8'],[0.75,'#756bb1'],[1,'#54278f']],
    'winter':    [[0,'#f7fcfd'],[0.25,'#99d8c9'],[0.5,'#41ae76'],[0.75,'#238b45'],[1,'#005824']],
    'deviation': [[0,'#1a9850'],[0.25,'#91cf60'],[0.5,'#ffffbf'],[0.75,'#fc8d59'],[1,'#d73027']],
    'multi':     [[0,'#fff5eb'],[0.25,'#fdd0a2'],[0.5,'#fd8d3c'],[0.75,'#d94801'],[1,'#7f2704']],
    'delta':     [[0,'#2166ac'],[0.25,'#92c5de'],[0.5,'#f7f7f7'],[0.75,'#f4a582'],[1,'#b2182b']],
}

# ---------------------------------------------------------------------------
# NAME NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_pred_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['county_upper'] = df['county'].str.upper().str.strip()
    df.loc[df['state'] == 'LA', 'county_upper'] = (
        df.loc[df['state'] == 'LA', 'county_upper'].str.replace(' PARISH', '', regex=False))
    va_mask    = df['state'] == 'VA'
    va_protect = df['county_upper'].isin(['CHARLES CITY', 'JAMES CITY'])
    df.loc[va_mask & ~va_protect, 'county_upper'] = (
        df.loc[va_mask & ~va_protect, 'county_upper'].str.replace(' CITY', '', regex=False))
    df.loc[df['state'] == 'CT', 'county_upper'] = (
        df.loc[df['state'] == 'CT', 'county_upper'].str.replace(' PLANNING REGION', '', regex=False))
    df.loc[(df['state'] == 'NM') & (df['county_upper'] == 'DOAA ANA'), 'county_upper'] = 'DOÑA ANA'
    df.loc[(df['state'] == 'MD') & (df['county_upper'] == 'BALTIMORE CITY'), 'county_upper'] = 'BALTIMORE'
    df.loc[(df['state'] == 'MO') & (df['county'] == 'St. Louis City'), 'county_upper'] = 'ST. LOUIS CITY'
    return df


def prepare_rapt(rapt: pd.DataFrame) -> pd.DataFrame:
    rapt = rapt.copy()
    rapt['state']        = rapt['State Name Abbreviation']
    rapt['county_upper'] = rapt['County Name'].str.upper().str.strip()
    stl_city = rapt[(rapt['state'] == 'MO') & (rapt['State-County FIPS Code'] == 29510)].copy()
    stl_city['county_upper'] = 'ST. LOUIS CITY'
    return pd.concat([rapt, stl_city], ignore_index=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_rapt(path: str = str(NRI_PATH)) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=NRI_COLS)
    return prepare_rapt(df)


@st.cache_data(ttl=3600)
def load_predictions(month: int) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"national_predictions_month{month:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return normalize_pred_names(pd.read_csv(path))


@st.cache_data(ttl=86400)
def load_nfip_summary() -> pd.DataFrame:
    """Load pre-aggregated NFIP county claim summary."""
    if not NFIP_SUMMARY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(NFIP_SUMMARY_PATH)
    df['countyCode'] = df['countyCode'].astype(str).str.replace('.0', '', regex=False).str.zfill(5)
    return df


def load_nfip_policies() -> pd.DataFrame:
    """Load pre-aggregated NFIP county policy summary (TIV, policy count, premiums)."""
    if not NFIP_POLICIES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(NFIP_POLICIES_PATH)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    return df


@st.cache_data(ttl=86400)
def load_climate_trends() -> dict:
    """Load flood severity trends and hazard frequency trends."""
    flood_trends = {}
    if FLOOD_TRENDS_PATH.exists():
        ft = pd.read_csv(FLOOD_TRENDS_PATH)
        flood_trends = dict(zip(ft['state'], ft['annual_severity_trend']))
    hazard_trends = {}
    if HAZARD_TRENDS_PATH.exists():
        ht = pd.read_csv(HAZARD_TRENDS_PATH)
        for _, row in ht.iterrows():
            key = (row['state'], row['hazard'])
            hazard_trends[key] = row['annual_freq_trend']
    return {'flood_severity': flood_trends, 'hazard_freq': hazard_trends}


@st.cache_data(ttl=86400)
def load_fema_pa() -> pd.DataFrame:
    """Load pre-aggregated FEMA PA county summary."""
    if not FEMA_PA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FEMA_PA_PATH)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    return df


@st.cache_data(ttl=86400)
def load_geojson() -> dict:
    """Load county GeoJSON from local file — same one AHI uses."""
    import json
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Missing: {GEOJSON_PATH}. Copy national_counties.geojson into data/.")
    with open(GEOJSON_PATH, "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# COMPUTATION
# ---------------------------------------------------------------------------

def merge_data(pred: pd.DataFrame, rapt: pd.DataFrame) -> pd.DataFrame:
    keep = ['state', 'county_upper', 'State-County FIPS Code',
            'Population (2020)', 'Building Value ($)',
            'Wildfire - Expected Annual Loss - Building Value',
            'Wildfire - Expected Annual Loss - Agriculture Value',
            'Inland Flooding - Expected Annual Loss - Building Value',
            'Strong Wind - Expected Annual Loss - Building Value',
            'Ice Storm - Expected Annual Loss - Building Value',
            'Hurricane - Expected Annual Loss - Building Value',
            'Expected Annual Loss - Building Value - Composite',
            'Expected Annual Loss - Agriculture Value - Composite',
            'Wildfire - Annualized Frequency',
            'Inland Flooding - Annualized Frequency',
            'Strong Wind - Annualized Frequency',
            'Ice Storm - Annualized Frequency',
            'Hurricane - Annualized Frequency',
            'Social Vulnerability - Score',
            'Social Vulnerability - Rating',
            'Community Resilience - Score',
            'Community Resilience - Rating']
    keep = [c for c in keep if c in rapt.columns]
    merged = pred.merge(rapt[keep], on=['state', 'county_upper'], how='left')
    merged['fips'] = merged['State-County FIPS Code'].astype(str).str.zfill(5)

    # Recompute max_hazard and max_p from only the 4 active hazards
    # Overrides CSV values which include seismic_p
    _label_map = {'fire_p': 'fire', 'flood_p': 'flood', 'wind_p': 'wind', 'winter_p': 'winter'}
    merged['max_p']      = merged[HAZARD_COLS].max(axis=1)
    merged['max_hazard'] = merged[HAZARD_COLS].idxmax(axis=1).map(_label_map)

    return merged


def _loss_given_event(eal: pd.Series, freq) -> pd.Series:
    """Derive mean loss per event: EAL / annualized_frequency.

    EAL already encodes probability (EAL = freq × mean_severity).
    Dividing out frequency isolates the pure severity component so we
    can re-weight by AHI's conditional probability without double-counting.
    Falls back to EAL when frequency is zero or missing (conservative).
    """
    if freq is None:
        return eal.fillna(0)
    safe_freq = freq.fillna(0).replace(0, np.nan)
    lge = eal / safe_freq
    return lge.fillna(eal.fillna(0))


def compute_cel(df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    df = df.copy()
    scalar = window_days / 365

    fire_eal = (df['Wildfire - Expected Annual Loss - Building Value'].fillna(0)
                + df['Wildfire - Expected Annual Loss - Agriculture Value'].fillna(0))
    fire_lge = _loss_given_event(fire_eal, df.get('Wildfire - Annualized Frequency'))

    flood_eal = df['Inland Flooding - Expected Annual Loss - Building Value'].fillna(0)
    flood_lge = _loss_given_event(flood_eal, df.get('Inland Flooding - Annualized Frequency'))

    # Combine Strong Wind + Hurricane EAL for wind hazard (AHI "wind" covers both)
    wind_eal = (df['Strong Wind - Expected Annual Loss - Building Value'].fillna(0)
                + df['Hurricane - Expected Annual Loss - Building Value'].fillna(0))
    wind_freq = (df.get('Strong Wind - Annualized Frequency', pd.Series(0, index=df.index)).fillna(0)
                 + df.get('Hurricane - Annualized Frequency', pd.Series(0, index=df.index)).fillna(0))
    wind_lge = _loss_given_event(wind_eal, wind_freq)

    winter_eal = df['Ice Storm - Expected Annual Loss - Building Value'].fillna(0)
    winter_lge = _loss_given_event(winter_eal, df.get('Ice Storm - Annualized Frequency'))

    # Cap LGE at total county building value (loss can't exceed exposure)
    bldg_val = df['Building Value ($)'].fillna(0)
    df['lge_fire']   = fire_lge.clip(upper=bldg_val)
    df['lge_flood']  = flood_lge.clip(upper=bldg_val)
    df['lge_wind']   = wind_lge.clip(upper=bldg_val)
    df['lge_winter'] = winter_lge.clip(upper=bldg_val)

    df['cel_fire']   = df['fire_p']   * fire_lge   * scalar
    df['cel_flood']  = df['flood_p']  * flood_lge  * scalar
    df['cel_wind']   = df['wind_p']   * wind_lge   * scalar
    df['cel_winter'] = df['winter_p'] * winter_lge * scalar
    df['cel_total']  = df[['cel_fire', 'cel_flood', 'cel_wind', 'cel_winter']].sum(axis=1)
    return df


def compute_deviations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h, b in HIST_BASELINE.items():
        df[f'dev_{h}'] = df[h] / b
    df['dev_max'] = df[[f'dev_{h}' for h in HAZARD_COLS]].max(axis=1)
    return df


def compute_multi_hazard_flag(df: pd.DataFrame,
                               threshold: float = MULTI_HAZARD_DEV_THRESHOLD,
                               min_count: int   = MULTI_HAZARD_MIN_COUNT) -> pd.DataFrame:
    df       = df.copy()
    dev_cols = [f'dev_{h}' for h in HAZARD_COLS]
    df['multi_hazard_count'] = (df[dev_cols] >= threshold).sum(axis=1)
    df['multi_hazard_flag']  = df['multi_hazard_count'] >= min_count
    df['elevated_hazards']   = df.apply(
        lambda row: ', '.join(HAZARD_LABELS[h] for h in HAZARD_COLS if row.get(f'dev_{h}', 0) >= threshold),
        axis=1
    )
    return df


def compute_svi_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """Weight CEL by Social Vulnerability Index (higher SVI = less capacity to absorb loss)."""
    df = df.copy()
    svi = df['Social Vulnerability - Score'].fillna(50)
    svi_weight = svi / 50  # normalized so SVI=50 (median) → weight=1.0
    df['svi_score'] = svi
    df['svi_weight'] = svi_weight.round(3)
    df['cel_svi_weighted'] = (df['cel_total'] * svi_weight).round(0)
    cr = df['Community Resilience - Score'].fillna(50)
    df['resilience_score'] = cr
    return df


def apply_credibility_weighting(df: pd.DataFrame, nfip: pd.DataFrame,
                                nfip_policies: pd.DataFrame = None,
                                k: int = CREDIBILITY_K) -> pd.DataFrame:
    """Blend NRI modeled flood LGE with actual NFIP claims experience.

    Uses Buhlmann credibility: Z = n / (n + k), where n = claim count.
    Counties with many claims → actual experience dominates.
    Counties with few claims → NRI model dominates.
    When policies data is available, also computes loss ratio (claims/TIV).
    """
    df = df.copy()
    if nfip.empty:
        df['flood_credibility'] = 0.0
        df['lge_flood_credibility'] = df['lge_flood']
        df['nfip_policy_count'] = 0
        df['nfip_tiv'] = 0.0
        df['nfip_loss_ratio'] = np.nan
        df['nfip_premium_total'] = 0.0
        return df

    nfip_sub = nfip[['countyCode', 'claim_count', 'avg_paid_per_claim', 'annual_paid', 'total_paid']].copy()
    nfip_sub = nfip_sub.rename(columns={'countyCode': 'fips'})
    df = df.merge(nfip_sub, on='fips', how='left')

    n = df['claim_count'].fillna(0)
    Z = n / (n + k)
    df['flood_credibility'] = Z.round(3)

    actual_lge = df['avg_paid_per_claim'].fillna(0)
    modeled_lge = df['lge_flood']
    df['lge_flood_credibility'] = (Z * actual_lge + (1 - Z) * modeled_lge).round(0)

    df['cel_flood_credibility'] = (
        df['flood_p'] * df['lge_flood_credibility'] * (DEFAULT_WINDOW_DAYS / 365)
    ).round(0)

    if nfip_policies is not None and not nfip_policies.empty:
        pol_sub = nfip_policies[['fips', 'policy_count', 'total_tiv',
                                 'total_premium', 'avg_premium']].copy()
        pol_sub = pol_sub.rename(columns={
            'policy_count': 'nfip_policy_count',
            'total_tiv': 'nfip_tiv',
            'total_premium': 'nfip_premium_total',
            'avg_premium': 'nfip_avg_premium',
        })
        df = df.merge(pol_sub, on='fips', how='left')
        df['nfip_policy_count'] = df['nfip_policy_count'].fillna(0)
        df['nfip_tiv'] = df['nfip_tiv'].fillna(0)
        df['nfip_premium_total'] = df['nfip_premium_total'].fillna(0)
        df['nfip_avg_premium'] = df['nfip_avg_premium'].fillna(0)
        safe_tiv = df['nfip_tiv'].replace(0, np.nan)
        df['nfip_loss_ratio'] = (df['annual_paid'].fillna(0) / safe_tiv).round(6)
    else:
        df['nfip_policy_count'] = 0
        df['nfip_tiv'] = 0.0
        df['nfip_loss_ratio'] = np.nan
        df['nfip_premium_total'] = 0.0

    return df


def apply_climate_trends(df: pd.DataFrame, trends: dict,
                         projection_years: int = 5) -> pd.DataFrame:
    """Adjust CEL for observed climate trends.

    Applies compound trend over projection_years to the loss-given-event.
    Flood: severity trend from NFIP claims (dollars per claim increasing).
    Fire/wind/winter: frequency trend from NOAA Storm Events.
    """
    df = df.copy()
    flood_trends = trends.get('flood_severity', {})
    freq_trends = trends.get('hazard_freq', {})

    multipliers = pd.Series(1.0, index=df.index)
    for haz, lge_col, cel_col in [
        ('fire', 'lge_fire', 'cel_fire'),
        ('flood', 'lge_flood', 'cel_flood'),
        ('wind', 'lge_wind', 'cel_wind'),
        ('winter', 'lge_winter', 'cel_winter'),
    ]:
        trend_col = f'trend_{haz}'
        if haz == 'flood':
            df[trend_col] = df['state'].map(flood_trends).fillna(0)
        else:
            df[trend_col] = df['state'].apply(
                lambda s, h=haz: freq_trends.get((s.upper(), h), 0)
            )
        multiplier = (1 + df[trend_col]) ** projection_years
        multiplier = multiplier.clip(0.5, 3.0)  # sanity cap
        df[f'{cel_col}_trend_adj'] = (df[cel_col] * multiplier).round(0)

    trend_cel_cols = [f'cel_{h}_trend_adj' for h in ['fire', 'flood', 'wind', 'winter']]
    df['cel_total_trend_adj'] = df[trend_cel_cols].sum(axis=1)
    return df


def apply_fema_pa(df: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Merge FEMA Public Assistance expenditure history into pipeline.

    Adds per-hazard and total PA obligated, annualized PA, and disaster
    counts. Computes model-vs-actual ratio (CEL / annual PA) as a
    ground-truth validation signal.
    """
    df = df.copy()
    if pa.empty:
        for h in ['fire', 'flood', 'wind', 'winter']:
            df[f'pa_annual_{h}'] = 0.0
            df[f'pa_total_{h}'] = 0.0
            df[f'pa_disasters_{h}'] = 0
        df['pa_annual_total'] = 0.0
        df['pa_total_total'] = 0.0
        df['pa_disasters_total'] = 0
        df['cel_pa_ratio'] = np.nan
        return df

    pa_cols = {
        'pa_annual_pa_fire': 'pa_annual_fire',
        'pa_annual_pa_flood': 'pa_annual_flood',
        'pa_annual_pa_wind': 'pa_annual_wind',
        'pa_annual_pa_winter': 'pa_annual_winter',
        'pa_annual_pa_total': 'pa_annual_total',
        'pa_total_obligated_fire': 'pa_total_fire',
        'pa_total_obligated_flood': 'pa_total_flood',
        'pa_total_obligated_wind': 'pa_total_wind',
        'pa_total_obligated_winter': 'pa_total_winter',
        'pa_total_obligated_total': 'pa_total_total',
        'pa_n_disasters_fire': 'pa_disasters_fire',
        'pa_n_disasters_flood': 'pa_disasters_flood',
        'pa_n_disasters_wind': 'pa_disasters_wind',
        'pa_n_disasters_winter': 'pa_disasters_winter',
        'pa_n_disasters_total': 'pa_disasters_total',
    }
    pa_sub = pa[['fips'] + list(pa_cols.keys())].copy()
    pa_sub = pa_sub.rename(columns=pa_cols)
    df = df.merge(pa_sub, on='fips', how='left')

    for col in pa_cols.values():
        df[col] = df[col].fillna(0)

    safe_pa = df['pa_annual_total'].replace(0, np.nan)
    df['cel_pa_ratio'] = (df['cel_total'] / safe_pa).round(2)

    return df


def compute_ep_curve(prob: float, mean_lge: float, cv: float = SEVERITY_CV,
                     n_points: int = 50) -> pd.DataFrame:
    """Compute exceedance probability curve for a single county-hazard pair.

    Assumes lognormal severity distribution (standard in property CAT modeling).
    Parameters derived from NRI loss-given-event and assumed CV.

    Returns DataFrame with columns: loss_threshold, exceedance_prob, return_period.
    """
    if mean_lge <= 0 or prob <= 0:
        return pd.DataFrame({'loss_threshold': [], 'exceedance_prob': [], 'return_period': []})

    sigma2 = np.log(1 + cv**2)
    mu = np.log(mean_lge) - sigma2 / 2
    sigma = np.sqrt(sigma2)

    max_loss = stats.lognorm.ppf(0.999, s=sigma, scale=np.exp(mu))
    thresholds = np.linspace(0, max_loss, n_points)

    # P(Loss > L) = P(event occurs) × P(severity > L | event)
    surv = prob * stats.lognorm.sf(thresholds, s=sigma, scale=np.exp(mu))
    surv = np.clip(surv, 1e-6, 1.0)

    return pd.DataFrame({
        'loss_threshold': thresholds,
        'exceedance_prob': surv,
        'return_period': 1.0 / surv,
    })


def compute_return_periods(df: pd.DataFrame,
                           return_years: list = None) -> pd.DataFrame:
    """Compute loss estimates at standard return periods for all counties.

    Return period = 1 / P(Loss > L in a given year).
    Uses AHI probability as annual event frequency and lognormal severity.
    """
    if return_years is None:
        return_years = [10, 25, 50, 100, 250]

    hazard_map = {
        'fire':   ('fire_p',   'lge_fire'),
        'flood':  ('flood_p',  'lge_flood'),
        'wind':   ('wind_p',   'lge_wind'),
        'winter': ('winter_p', 'lge_winter'),
    }

    for rp in return_years:
        target_ep = 1.0 / rp
        for haz, (prob_col, lge_col) in hazard_map.items():
            col_name = f'rp{rp}_{haz}'
            losses = []
            for _, row in df.iterrows():
                p = row.get(prob_col, 0)
                lge = row.get(lge_col, 0)
                if p <= 0 or lge <= 0 or p < target_ep:
                    losses.append(0.0)
                    continue
                sigma2 = np.log(1 + SEVERITY_CV**2)
                mu = np.log(lge) - sigma2 / 2
                sigma = np.sqrt(sigma2)
                # Solve: p × SF(L) = target_ep → SF(L) = target_ep / p
                sf_target = target_ep / p
                if sf_target >= 1.0:
                    losses.append(0.0)
                else:
                    L = stats.lognorm.isf(sf_target, s=sigma, scale=np.exp(mu))
                    losses.append(max(0.0, L))
            df[col_name] = losses

        # Total across hazards for this return period
        haz_cols = [f'rp{rp}_{h}' for h in hazard_map]
        df[f'rp{rp}_total'] = df[haz_cols].sum(axis=1)

    return df


def compute_month_delta(df_a: pd.DataFrame, df_b: pd.DataFrame,
                         label_a: str = "June", label_b: str = "July") -> pd.DataFrame:
    join_cols = ['state', 'county_upper', 'county', 'county_id', 'fips']
    a = df_a[join_cols + HAZARD_COLS + [f'dev_{h}' for h in HAZARD_COLS]].copy()
    b = df_b[join_cols + HAZARD_COLS + [f'dev_{h}' for h in HAZARD_COLS]].copy()
    a = a.rename(columns={h: f'{h}_{label_a}' for h in HAZARD_COLS} |
                          {f'dev_{h}': f'dev_{h}_{label_a}' for h in HAZARD_COLS})
    b = b.rename(columns={h: f'{h}_{label_b}' for h in HAZARD_COLS} |
                          {f'dev_{h}': f'dev_{h}_{label_b}' for h in HAZARD_COLS})
    merged = a.merge(b[['state','county_upper'] +
                        [f'{h}_{label_b}' for h in HAZARD_COLS] +
                        [f'dev_{h}_{label_b}' for h in HAZARD_COLS]],
                     on=['state','county_upper'], how='inner')
    for h in HAZARD_COLS:
        merged[f'delta_{h}']     = merged[f'{h}_{label_b}']     - merged[f'{h}_{label_a}']
        merged[f'delta_dev_{h}'] = merged[f'dev_{h}_{label_b}'] - merged[f'dev_{h}_{label_a}']
    delta_cols = [f'delta_{h}' for h in HAZARD_COLS]
    merged['max_spike']    = merged[delta_cols].max(axis=1)
    merged['max_drop']     = merged[delta_cols].min(axis=1)
    merged['spike_hazard'] = merged[delta_cols].idxmax(axis=1).str.replace('delta_','').str.replace('_p','').str.title()
    merged['drop_hazard']  = merged[delta_cols].idxmin(axis=1).str.replace('delta_','').str.replace('_p','').str.title()
    return merged


def full_pipeline(month: int, rapt: pd.DataFrame,
                  window_days: int = DEFAULT_WINDOW_DAYS,
                  mh_threshold: float = MULTI_HAZARD_DEV_THRESHOLD) -> pd.DataFrame:
    pred   = load_predictions(month)
    merged = merge_data(pred, rapt)
    merged = compute_cel(merged, window_days)
    merged = compute_deviations(merged)
    merged = compute_multi_hazard_flag(merged, threshold=mh_threshold)
    merged = compute_svi_weighted(merged)
    nfip = load_nfip_summary()
    nfip_policies = load_nfip_policies()
    merged = apply_credibility_weighting(merged, nfip, nfip_policies)
    trends = load_climate_trends()
    merged = apply_climate_trends(merged, trends)
    pa = load_fema_pa()
    merged = apply_fema_pa(merged, pa)
    return merged

# ---------------------------------------------------------------------------
# MAP HELPERS
# ---------------------------------------------------------------------------

def make_choropleth(df: pd.DataFrame, color_col: str, title: str,
                    geojson: dict = None,
                    color_scale: str = 'cel',
                    hover_cols: list = None,
                    range_color: tuple = None,
                    map_style: str = 'Dark') -> go.Figure:
    """
    Build a Plotly Choroplethmapbox using the same GeoJSON and _id join
    convention as AHI (properties._id = state|county_id).
    """
    df = df.copy()
    # Build _id join key matching AHI's GeoJSON properties._id
    df['_id'] = df['state'] + '|' + df['county_id']

    # Numeric values for color
    z = df[color_col]
    if range_color is None:
        range_color = (z.quantile(0.02), z.quantile(0.98))

    # Hover text — format z based on metric type
    if 'cel' in color_col:
        z_fmt = '$%{z:,.0f}'
    elif 'dev' in color_col or color_col == 'multi_hazard_count':
        z_fmt = '%{z:.2f}×'
    elif 'delta' in color_col:
        z_fmt = '%{z:+.1%}'
    elif '_p' in color_col:
        z_fmt = '%{z:.1%}'
    else:
        z_fmt = '%{z:.2f}'

    # Friendly column label map for hover
    _HOVER_LABELS = {
        'state': 'State', 'cel_total': 'Total CEL', 'cel_fire': 'Fire CEL',
        'cel_flood': 'Flood CEL', 'cel_wind': 'Wind CEL', 'cel_winter': 'Winter CEL',
        'max_hazard': 'Dominant Hazard', 'dev_fire_p': 'Fire Dev', 'dev_flood_p': 'Flood Dev',
        'dev_wind_p': 'Wind Dev', 'dev_winter_p': 'Winter Dev', 'dev_max': 'Max Dev',
        'elevated_hazards': 'Elevated', 'spike_hazard': 'Spike Hazard',
        'drop_hazard': 'Drop Hazard', 'county': 'County',
    }

    # Pre-format extra hover columns so values are readable in tooltip
    extra_cols = hover_cols or []
    hover_df = df[['county', 'state'] + extra_cols].copy()
    _dollar_cols = {'cel_total', 'cel_fire', 'cel_flood', 'cel_wind', 'cel_winter',
                    'cel_flood_credibility', 'cel_total_trend_adj',
                    'pa_annual_total', 'nfip_tiv', 'nfip_premium_total'}
    for col in extra_cols:
        if col in _dollar_cols:
            hover_df[col] = hover_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        elif col.startswith('dev_'):
            hover_df[col] = hover_df[col].apply(lambda x: f"{x:.2f}×" if pd.notna(x) else "—")
        elif col.endswith('_p'):
            hover_df[col] = hover_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        elif col == 'nfip_loss_ratio':
            hover_df[col] = hover_df[col].apply(lambda x: f"{x:.3%}" if pd.notna(x) else "—")
        elif col in ('nfip_policy_count', 'pa_disasters_total'):
            hover_df[col] = hover_df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    customdata = hover_df.values

    hover_lines = '<b>%{customdata[0]}, %{customdata[1]}</b><br>'
    hover_lines += f'{title}: {z_fmt}<br>'
    for i, col in enumerate(extra_cols):
        label = _HOVER_LABELS.get(col, col.replace('_',' ').title())
        hover_lines += f'{label}: %{{customdata[{i+2}]}}<br>'
    hover_lines += '<extra></extra>'

    style      = _TILE_STYLES.get(map_style, _TILE_STYLES['Dark'])
    colorscale = _COLOR_SCALES.get(color_scale, _COLOR_SCALES['cel'])

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=df['_id'],
        z=z,
        featureidkey='properties._id',
        colorscale=colorscale,
        zmin=range_color[0],
        zmax=range_color[1],
        marker_line_width=0.5,
        marker_line_color=style['border'],
        marker_opacity=style['opacity'],
        customdata=customdata,
        hovertemplate=hover_lines,
        colorbar=dict(
            thickness=12, len=0.5,
            title=dict(text=title, side='right', font=dict(size=11)),
            tickformat='$,.0f' if 'cel' in color_col else (
                '.1%' if '_p' in color_col or 'delta' in color_col else '.1f'
            ),
        ),
    ))

    fig.update_layout(
        mapbox_style=style['mapbox_style'],
        mapbox_layers=style['mapbox_layers'],
        mapbox_zoom=3.2,
        mapbox_center={'lat': 38.5, 'lon': -96},
        margin={'r': 0, 't': 35, 'l': 0, 'b': 0},
        height=540,
        title=dict(text=title, font=dict(size=14, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    return fig


def show_map_and_table(fig, table_df: pd.DataFrame, table_height: int = 400):
    """Render map tab + table tab side by side using st.tabs."""
    map_tab, tbl_tab = st.tabs(["🗺️ Map", "📋 Table"])
    with map_tab:
        st.plotly_chart(fig, use_container_width=True)
    with tbl_tab:
        st.dataframe(table_df, use_container_width=True, height=table_height)

# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AHI Exposure Intelligence", page_icon="📊", layout="wide")
    st.title("AHI Exposure Intelligence")
    st.caption(
        "Conditional Expected Loss (CEL) = AHI probability × FEMA NRI Expected Annual Loss "
        "× forecast window. Decision-support only — not actuarial output. "
        "Dollar figures from FEMA NRI/RAPT (2024 release)."
    )

    with st.sidebar:
        st.header("Controls")
        month        = st.selectbox("Forecast month", list(AVAILABLE_MONTHS.keys()),
                                    format_func=lambda m: AVAILABLE_MONTHS[m])
        window       = st.slider("Forecast window (days)", 7, 30, DEFAULT_WINDOW_DAYS)
        state_filter = st.text_input("Filter by state (2-letter)", "").upper().strip()
        mh_threshold = st.slider("Multi-hazard deviation threshold", 1.0, 5.0,
                                 float(MULTI_HAZARD_DEV_THRESHOLD), 0.25)
        map_style = st.selectbox("Map style", ["Dark", "Light", "Satellite"])
        st.divider()
        view_mode = st.radio("View", [
            "Portfolio Summary",
            "Conditional Expected Loss",
            "NFIP Credibility",
            "Climate Trend Adjustment",
            "FEMA PA Validation",
            "Exceedance Probability",
            "Vulnerability-Weighted",
            "Deviation from Baseline",
            "Multi-Hazard Flags",
            "Month-over-Month Delta",
        ])
        st.divider()
        st.caption("Data: FEMA NRI/RAPT · AHI v4.0 · RAL LLC")

    try:
        rapt = load_rapt()
    except FileNotFoundError:
        st.error(f"NRI data not found at `{NRI_PATH}`.")
        st.stop()

    try:
        geojson = load_geojson()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        result = full_pipeline(month, rapt, window, mh_threshold)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if state_filter and len(state_filter) == 2:
        result = result[result['state'] == state_filter]
        if result.empty:
            st.warning(f"No data for state '{state_filter}'")
            st.stop()

    if view_mode == "Portfolio Summary":
        _render_portfolio(result, window, month, geojson, map_style)
    elif view_mode == "Conditional Expected Loss":
        _render_cel(result, window, geojson, map_style)
    elif view_mode == "NFIP Credibility":
        _render_credibility(result, geojson, map_style)
    elif view_mode == "Climate Trend Adjustment":
        _render_climate_trend(result, geojson, map_style)
    elif view_mode == "FEMA PA Validation":
        _render_fema_pa(result, geojson, map_style)
    elif view_mode == "Exceedance Probability":
        _render_ep(result, geojson, map_style)
    elif view_mode == "Vulnerability-Weighted":
        _render_svi(result, geojson, map_style)
    elif view_mode == "Deviation from Baseline":
        _render_deviation(result, geojson, map_style)
    elif view_mode == "Multi-Hazard Flags":
        _render_multi_hazard(result, mh_threshold, geojson, map_style)
    elif view_mode == "Month-over-Month Delta":
        _render_delta(rapt, window, state_filter, geojson, map_style)


# ---------------------------------------------------------------------------
# VIEW RENDERERS
# ---------------------------------------------------------------------------

def _render_portfolio(df: pd.DataFrame, window: int, month: int, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Portfolio Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CEL (all counties)", f"${df['cel_total'].sum():,.0f}")
    c2.metric("Fire CEL",   f"${df['cel_fire'].sum():,.0f}")
    c3.metric("Flood CEL",  f"${df['cel_flood'].sum():,.0f}")
    c4.metric("Wind CEL",   f"${df['cel_wind'].sum():,.0f}")
    st.divider()

    # Map selector
    map_metric = st.selectbox("Map metric", [
        "Total CEL", "Fire CEL", "Flood CEL", "Wind CEL", "Winter CEL"
    ])
    col_map  = {"Total CEL":"cel_total","Fire CEL":"cel_fire","Flood CEL":"cel_flood",
                "Wind CEL":"cel_wind","Winter CEL":"cel_winter"}
    cscale   = {"Total CEL":"cel","Fire CEL":"fire","Flood CEL":"flood",
                "Wind CEL":"wind","Winter CEL":"winter"}
    col      = col_map[map_metric]
    hover    = ['state', col, 'max_hazard']
    fig      = make_choropleth(df, col, f"{map_metric} — {AVAILABLE_MONTHS[month]}",
                               geojson=geojson, color_scale=cscale[map_metric], hover_cols=hover, map_style=map_style)

    # Top 20 table
    top20 = df.nlargest(20, 'cel_total')[['state','county','cel_total','max_hazard',
                                          'Population (2020)','multi_hazard_flag',
                                          'elevated_hazards']].copy()
    top20['cel_total']         = top20['cel_total'].apply(lambda x: f"${x:,.0f}")
    top20['Population (2020)'] = top20['Population (2020)'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    top20['multi_hazard_flag'] = top20['multi_hazard_flag'].apply(lambda x: "⚠️" if x else "")
    top20.columns = ['State','County','CEL Total','Dominant Hazard','Population','Multi⚠️','Elevated Hazards']

    show_map_and_table(fig, top20)

    st.divider()
    st.subheader("CEL by State (aggregate)")
    by_state = df.groupby('state')['cel_total'].sum().sort_values(ascending=False).reset_index()
    by_state['cel_total'] = by_state['cel_total'].apply(lambda x: f"${x:,.0f}")
    by_state.columns = ['State','Total CEL']
    st.dataframe(by_state, use_container_width=True, height=350)
    st.caption(f"Window: {window} days · Not actuarial output.")


def _render_cel(df: pd.DataFrame, window: int, geojson: dict = None, map_style: str = "Dark"):
    st.subheader(f"Conditional Expected Loss — {window}-day window")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CEL", f"${df['cel_total'].sum():,.0f}")
    c2.metric("Highest county", f"${df['cel_total'].max():,.0f}")
    c3.metric("Counties > $1M CEL", int((df['cel_total'] >= 1_000_000).sum()))
    c4.metric("Matched counties", int(df['State-County FIPS Code'].notna().sum()))

    sort_by   = st.selectbox("Sort / map by", ["CEL Total","CEL Fire","CEL Flood","CEL Wind","CEL Winter"])
    sort_map  = {"CEL Total":"cel_total","CEL Fire":"cel_fire","CEL Flood":"cel_flood",
                 "CEL Wind":"cel_wind","CEL Winter":"cel_winter"}
    cscale_map= {"CEL Total":"RdYlGn_r","CEL Fire":"YlOrRd","CEL Flood":"Blues",
                 "CEL Wind":"Purples","CEL Winter":"ice"}
    col       = sort_map[sort_by]

    fig = make_choropleth(df, col, sort_by, geojson=geojson, color_scale=cscale_map[sort_by],
                          hover_cols=['state','cel_total','max_hazard'], map_style=map_style)

    display = df[['state','county','fire_p','wind_p','flood_p','winter_p',
                  'cel_fire','cel_wind','cel_flood','cel_winter','cel_total',
                  'multi_hazard_flag','elevated_hazards']].copy()
    display = display.sort_values(col, ascending=False)
    for c in ['fire_p','wind_p','flood_p','winter_p']:
        display[c] = display[c].apply(lambda x: f"{x:.1%}")
    for c in ['cel_fire','cel_wind','cel_flood','cel_winter','cel_total']:
        display[c] = display[c].apply(lambda x: f"${x:,.0f}")
    display['multi_hazard_flag'] = display['multi_hazard_flag'].apply(lambda x: "⚠️" if x else "")
    display.columns = ['State','County','Fire %','Wind %','Flood %','Winter %',
                       'CEL Fire','CEL Wind','CEL Flood','CEL Winter','CEL Total','Multi⚠️','Elevated']

    show_map_and_table(fig, display)
    st.caption("⚠️ Cross-reference NWS watches/warnings before operational action.")


def _render_credibility(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("NFIP Credibility-Weighted Flood Losses")
    st.caption(
        "Blends NRI modeled flood losses with actual NFIP claim experience using "
        f"Buhlmann credibility (Z = n / (n + {CREDIBILITY_K})). Counties with extensive "
        "claim history get actual experience; data-sparse counties default to NRI model."
    )

    has_nfip = df['flood_credibility'].sum() > 0
    if not has_nfip:
        st.warning("NFIP claims data not loaded. Place nfip_county_summary.csv in data/.")
        return

    has_policies = df['nfip_tiv'].sum() > 0

    c1, c2, c3, c4 = st.columns(4)
    high_cred = (df['flood_credibility'] >= 0.5).sum()
    c1.metric("Counties with NFIP data", int((df['claim_count'].fillna(0) > 0).sum()))
    c2.metric("High-credibility (Z >= 0.5)", int(high_cred))
    if has_policies:
        c3.metric("Total Insured Value", f"${df['nfip_tiv'].sum():,.0f}")
        with_lr = df[df['nfip_loss_ratio'].notna() & (df['nfip_loss_ratio'] > 0)]
        c4.metric("Median Loss Ratio", f"{with_lr['nfip_loss_ratio'].median():.3%}")
    else:
        c3.metric("Modeled Flood CEL", f"${df['cel_flood'].sum():,.0f}")
        c4.metric("Credibility-Adj Flood CEL", f"${df['cel_flood_credibility'].sum():,.0f}")

    metric_options = ["Credibility-Adjusted Flood CEL", "Credibility Weight (Z)",
                      "NFIP Claim Count", "Avg Paid per Claim"]
    col_map = {"Credibility-Adjusted Flood CEL": "cel_flood_credibility",
               "Credibility Weight (Z)": "flood_credibility",
               "NFIP Claim Count": "claim_count",
               "Avg Paid per Claim": "avg_paid_per_claim"}
    if has_policies:
        metric_options.extend(["NFIP Loss Ratio", "NFIP Policy Count",
                               "Total Insured Value", "Avg Premium"])
        col_map.update({"NFIP Loss Ratio": "nfip_loss_ratio",
                        "NFIP Policy Count": "nfip_policy_count",
                        "Total Insured Value": "nfip_tiv",
                        "Avg Premium": "nfip_avg_premium"})

    map_metric = st.selectbox("Map metric", metric_options)
    col = col_map[map_metric]
    if 'flood' in col or 'cel' in col:
        cscale = 'flood'
    elif 'loss_ratio' in col:
        cscale = 'deviation'
    else:
        cscale = 'cel'

    hover = ['state', 'county', 'flood_credibility', 'cel_flood', 'cel_flood_credibility']
    if has_policies:
        hover.extend(['nfip_policy_count', 'nfip_loss_ratio'])
    fig = make_choropleth(df, col, map_metric,
                          geojson=geojson, color_scale=cscale,
                          hover_cols=hover, map_style=map_style)

    # Table with loss ratio if policies data available
    tbl_cols = ['state', 'county', 'flood_p', 'lge_flood', 'cel_flood',
                'claim_count', 'avg_paid_per_claim', 'flood_credibility',
                'lge_flood_credibility', 'cel_flood_credibility']
    header_names = ['State', 'County', 'Flood P', 'NRI LGE', 'NRI CEL',
                    'NFIP Claims', 'NFIP Avg Paid', 'Z', 'Blended LGE', 'Blended CEL']
    if has_policies:
        tbl_cols.extend(['nfip_policy_count', 'nfip_loss_ratio'])
        header_names.extend(['Policies', 'Loss Ratio'])

    tbl = df[df['claim_count'].fillna(0) > 0].nlargest(50, 'cel_flood_credibility')[tbl_cols].copy()
    tbl['flood_p'] = tbl['flood_p'].apply(lambda x: f"{x:.1%}")
    for c in ['lge_flood', 'cel_flood', 'avg_paid_per_claim', 'lge_flood_credibility', 'cel_flood_credibility']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    tbl['claim_count'] = tbl['claim_count'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    tbl['flood_credibility'] = tbl['flood_credibility'].apply(lambda x: f"{x:.2f}")
    if has_policies:
        tbl['nfip_policy_count'] = tbl['nfip_policy_count'].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "—")
        tbl['nfip_loss_ratio'] = tbl['nfip_loss_ratio'].apply(
            lambda x: f"{x:.3%}" if pd.notna(x) else "—")
    tbl.columns = header_names

    show_map_and_table(fig, tbl)
    src = f"Z = n/(n+{CREDIBILITY_K}). Z=0 = pure NRI model. Z=1 = pure NFIP experience. "
    src += "Claims: 1978-2026, OpenFEMA."
    if has_policies:
        src += " Policies: 2025 in-force snapshot, ~4.6M policies, $1.3T TIV."
    st.caption(src)


def _render_climate_trend(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Climate Trend Adjustment")
    st.caption(
        "Projects current CEL forward using observed trends. "
        "Flood: severity trend from NFIP claims ($/claim). "
        "Fire/wind/winter: frequency trend from NOAA Storm Events."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CEL", f"${df['cel_total'].sum():,.0f}")
    c2.metric("5-yr Projected CEL", f"${df['cel_total_trend_adj'].sum():,.0f}")
    pct_change = (df['cel_total_trend_adj'].sum() / max(df['cel_total'].sum(), 1) - 1) * 100
    c3.metric("Portfolio Δ", f"{pct_change:+.1f}%")
    c4.metric("Projection Window", "5 years")

    map_metric = st.selectbox("Map metric", [
        "5-yr Projected CEL", "Fire Trend-Adj CEL", "Flood Trend-Adj CEL",
        "Wind Trend-Adj CEL", "Winter Trend-Adj CEL"])
    col_map = {"5-yr Projected CEL": "cel_total_trend_adj",
               "Fire Trend-Adj CEL": "cel_fire_trend_adj",
               "Flood Trend-Adj CEL": "cel_flood_trend_adj",
               "Wind Trend-Adj CEL": "cel_wind_trend_adj",
               "Winter Trend-Adj CEL": "cel_winter_trend_adj"}
    col = col_map[map_metric]
    fig = make_choropleth(df, col, map_metric,
                          geojson=geojson, color_scale='cel',
                          hover_cols=['state', 'county', 'cel_total', 'cel_total_trend_adj', 'max_hazard'],
                          map_style=map_style)

    # Per-hazard trend comparison table
    tbl = df.nlargest(30, 'cel_total_trend_adj')[
        ['state', 'county', 'cel_total', 'cel_total_trend_adj',
         'trend_fire', 'trend_flood', 'trend_wind', 'trend_winter']
    ].copy()
    tbl['cel_total'] = tbl['cel_total'].apply(lambda x: f"${x:,.0f}")
    tbl['cel_total_trend_adj'] = tbl['cel_total_trend_adj'].apply(lambda x: f"${x:,.0f}")
    for h in ['fire', 'flood', 'wind', 'winter']:
        tbl[f'trend_{h}'] = tbl[f'trend_{h}'].apply(lambda x: f"{x:+.1%}/yr" if pd.notna(x) else "—")
    tbl.columns = ['State', 'County', 'Current CEL', '5-yr Projected',
                   'Fire Trend', 'Flood Trend', 'Wind Trend', 'Winter Trend']

    show_map_and_table(fig, tbl)
    st.caption(
        "Trends from: NFIP claims 1995-2025 (flood severity), "
        "NOAA Storm Events 2000-2024 (fire/wind/winter frequency). "
        "Capped at 0.5×–3.0× to prevent extreme extrapolation."
    )


def _render_fema_pa(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("FEMA Public Assistance — Model vs. Actual")
    st.caption(
        "Compares AHI-modeled Conditional Expected Loss (CEL) against actual FEMA PA "
        "expenditure history (1998-2026). High CEL/PA ratio → model predicts more loss "
        "than historical spend. Low ratio → county has received more PA than model expects."
    )

    has_pa = df['pa_annual_total'].sum() > 0
    if not has_pa:
        st.warning("FEMA PA data not loaded. Place fema_pa_county_summary.csv in data/.")
        return

    with_pa = df[df['pa_annual_total'] > 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Counties with PA history", f"{len(with_pa):,}")
    c2.metric("Total PA Obligated", f"${df['pa_total_total'].sum():,.0f}")
    c3.metric("Total Annual PA", f"${df['pa_annual_total'].sum():,.0f}")
    c4.metric("Total Annualized CEL", f"${(df['cel_total'] * 365 / DEFAULT_WINDOW_DAYS).sum():,.0f}")

    view_sub = st.selectbox("Map metric", [
        "Total PA Obligated", "Annualized PA", "Disaster Count",
        "CEL / PA Ratio", "Fire PA", "Flood PA", "Wind PA", "Winter PA"])
    col_map = {
        "Total PA Obligated": "pa_total_total",
        "Annualized PA": "pa_annual_total",
        "Disaster Count": "pa_disasters_total",
        "CEL / PA Ratio": "cel_pa_ratio",
        "Fire PA": "pa_annual_fire",
        "Flood PA": "pa_annual_flood",
        "Wind PA": "pa_annual_wind",
        "Winter PA": "pa_annual_winter",
    }
    col = col_map[view_sub]
    if col == 'cel_pa_ratio':
        cscale = 'deviation'
    elif 'disasters' in col:
        cscale = 'multi'
    else:
        cscale = 'cel'

    fig = make_choropleth(df, col, view_sub,
                          geojson=geojson, color_scale=cscale,
                          hover_cols=['state', 'county', 'pa_annual_total', 'cel_total',
                                      'pa_disasters_total', 'max_hazard'],
                          map_style=map_style)

    # Per-hazard comparison table
    tbl = with_pa.nlargest(40, 'pa_annual_total')[
        ['state', 'county', 'cel_total', 'pa_annual_total', 'cel_pa_ratio',
         'pa_disasters_total', 'pa_annual_fire', 'pa_annual_flood',
         'pa_annual_wind', 'pa_annual_winter']
    ].copy()
    for c in ['cel_total', 'pa_annual_total', 'pa_annual_fire', 'pa_annual_flood',
              'pa_annual_wind', 'pa_annual_winter']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "—")
    tbl['pa_disasters_total'] = tbl['pa_disasters_total'].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else "—")
    tbl['cel_pa_ratio'] = tbl['cel_pa_ratio'].apply(
        lambda x: f"{x:.2f}×" if pd.notna(x) else "—")
    tbl.columns = ['State', 'County', 'CEL (14d)', 'Annual PA',
                   'CEL/PA', 'Disasters', 'Fire PA', 'Flood PA',
                   'Wind PA', 'Winter PA']

    show_map_and_table(fig, tbl)

    with st.expander("Interpretation guide"):
        st.markdown("""
**CEL/PA ratio** compares the model's annualized conditional expected loss to actual FEMA Public Assistance expenditure:

| Ratio | Interpretation |
|-------|---------------|
| **< 0.5×** | Model under-predicts relative to historical PA. County may have unique exposure drivers not captured in NRI. |
| **0.5–2.0×** | Reasonable agreement between modeled and actual. |
| **> 2.0×** | Model predicts higher loss than historical PA. May reflect underinsured exposure, unspent eligibility, or hazard shift. |

**Caveats**: PA is reimbursement-based (applicant must request), excludes private-sector losses, and is heavily skewed by major disaster declarations. CEL includes all building exposure, not just public infrastructure.
""")

    st.caption(
        "Source: OpenFEMA Public Assistance Funded Projects Details (1998-2026). "
        "811K projects, $280B total obligated. Incident types mapped to AHI hazards."
    )


def _render_ep(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Exceedance Probability & Return Periods")
    st.caption(
        "Standard CAT model output: loss estimates at various return periods. "
        "Severity modeled as lognormal (CV=1.5). "
        "Return period = 1 / P(Loss > threshold in a year)."
    )

    # Compute return periods for all counties
    rp_df = compute_return_periods(df.copy())
    rp_years = [10, 25, 50, 100, 250]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("100-yr Total Loss", f"${rp_df['rp100_total'].sum():,.0f}")
    c2.metric("250-yr Total Loss", f"${rp_df['rp250_total'].sum():,.0f}")
    c3.metric("Counties > $1M at 100-yr", int((rp_df['rp100_total'] >= 1_000_000).sum()))
    c4.metric("Counties > $1M at 250-yr", int((rp_df['rp250_total'] >= 1_000_000).sum()))

    # Map: 100-yr total loss
    rp_map = st.selectbox("Map return period", ["100-yr", "250-yr", "50-yr", "25-yr", "10-yr"])
    rp_col = {'100-yr': 'rp100_total', '250-yr': 'rp250_total', '50-yr': 'rp50_total',
              '25-yr': 'rp25_total', '10-yr': 'rp10_total'}[rp_map]
    fig = make_choropleth(rp_df, rp_col, f"{rp_map} Return Period Loss",
                          geojson=geojson, color_scale='cel',
                          hover_cols=['state', 'county', 'cel_total', 'max_hazard'],
                          map_style=map_style)

    # Return period table
    tbl = rp_df.nlargest(50, 'rp100_total')[
        ['state', 'county', 'cel_total'] +
        [f'rp{y}_total' for y in rp_years] +
        ['max_hazard']
    ].copy()
    tbl['cel_total'] = tbl['cel_total'].apply(lambda x: f"${x:,.0f}")
    for y in rp_years:
        tbl[f'rp{y}_total'] = tbl[f'rp{y}_total'].apply(lambda x: f"${x:,.0f}")
    tbl.columns = ['State', 'County', 'CEL'] + [f'{y}-yr' for y in rp_years] + ['Dominant']

    show_map_and_table(fig, tbl)

    # EP curve for a selected county
    st.divider()
    st.markdown("### Single-County EP Curve")
    ep_cols = st.columns([2, 1])
    with ep_cols[1]:
        ep_hazard = st.selectbox("Hazard", ["Fire", "Flood", "Wind", "Winter"], key='ep_haz')
    with ep_cols[0]:
        county_options = df.sort_values('cel_total', ascending=False)['county'].tolist()
        ep_county = st.selectbox("County", county_options[:200], key='ep_county')

    row = df[df['county'] == ep_county].iloc[0] if len(df[df['county'] == ep_county]) > 0 else None
    if row is not None:
        prob_col = {'Fire': 'fire_p', 'Flood': 'flood_p', 'Wind': 'wind_p', 'Winter': 'winter_p'}[ep_hazard]
        lge_col = {'Fire': 'lge_fire', 'Flood': 'lge_flood', 'Wind': 'lge_wind', 'Winter': 'lge_winter'}[ep_hazard]
        ep_data = compute_ep_curve(row[prob_col], row[lge_col])

        if not ep_data.empty:
            fig_ep = go.Figure()
            fig_ep.add_trace(go.Scatter(
                x=ep_data['loss_threshold'], y=ep_data['exceedance_prob'],
                mode='lines', fill='tozeroy',
                line=dict(color='#fd8d3c', width=2),
                fillcolor='rgba(253,141,60,0.15)',
                hovertemplate='Loss > $%{x:,.0f}<br>P = %{y:.4f}<br>Return: %{customdata:.0f} yr<extra></extra>',
                customdata=ep_data['return_period'],
            ))
            fig_ep.update_layout(
                title=f"{ep_hazard} EP Curve — {ep_county}, {row['state']}",
                xaxis_title="Loss Threshold ($)",
                yaxis_title="Exceedance Probability",
                yaxis_type="log", yaxis_range=[-4, 0],
                xaxis_tickformat="$,.0f",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            )
            # Add return period reference lines
            for rp, color in [(10, '#91cf60'), (100, '#fc8d59'), (250, '#d73027')]:
                fig_ep.add_hline(y=1/rp, line_dash="dash", line_color=color,
                                 annotation_text=f"{rp}-yr", annotation_position="right")
            st.plotly_chart(fig_ep, use_container_width=True)
            st.caption(
                f"AHI probability: {row[prob_col]:.1%} · "
                f"Mean loss given event: ${row[lge_col]:,.0f} · "
                f"Severity CV: {SEVERITY_CV}"
            )
        else:
            st.info(f"No {ep_hazard.lower()} exposure data for {ep_county}.")


def _render_svi(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Vulnerability-Weighted Exposure")
    st.caption(
        "CEL weighted by FEMA Social Vulnerability Index (SVI). "
        "Higher SVI = less community capacity to absorb loss. "
        "SVI=50 (median) → weight=1.0, SVI=100 → weight=2.0."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw CEL Total", f"${df['cel_total'].sum():,.0f}")
    c2.metric("SVI-Weighted CEL", f"${df['cel_svi_weighted'].sum():,.0f}")
    c3.metric("Amplification", f"{df['cel_svi_weighted'].sum() / max(df['cel_total'].sum(), 1):.2f}×")
    c4.metric("High-SVI Counties (>75th)", int((df['svi_score'] >= 75).sum()))

    map_metric = st.selectbox("Map metric", ["SVI-Weighted CEL", "SVI Score", "Community Resilience"])
    col_map = {"SVI-Weighted CEL": "cel_svi_weighted", "SVI Score": "svi_score",
               "Community Resilience": "resilience_score"}
    col = col_map[map_metric]
    cscale = 'cel' if 'cel' in col else 'deviation'

    fig = make_choropleth(df, col, map_metric,
                          geojson=geojson, color_scale=cscale,
                          hover_cols=['state', 'county', 'cel_total', 'svi_score', 'max_hazard'],
                          map_style=map_style)

    display = df.nlargest(50, 'cel_svi_weighted')[
        ['state', 'county', 'cel_total', 'svi_score', 'svi_weight',
         'cel_svi_weighted', 'resilience_score', 'max_hazard',
         'Population (2020)']
    ].copy()
    display['cel_total'] = display['cel_total'].apply(lambda x: f"${x:,.0f}")
    display['cel_svi_weighted'] = display['cel_svi_weighted'].apply(lambda x: f"${x:,.0f}")
    display['svi_score'] = display['svi_score'].apply(lambda x: f"{x:.1f}")
    display['svi_weight'] = display['svi_weight'].apply(lambda x: f"{x:.2f}×")
    display['resilience_score'] = display['resilience_score'].apply(lambda x: f"{x:.1f}")
    display['Population (2020)'] = display['Population (2020)'].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    display.columns = ['State', 'County', 'Raw CEL', 'SVI', 'SVI Weight',
                       'Weighted CEL', 'Resilience', 'Dominant', 'Population']

    show_map_and_table(fig, display)
    st.caption(
        "SVI source: CDC/ATSDR via FEMA NRI. "
        "Resilience source: FEMA Community Resilience estimates. "
        "Equity-weighted view for prioritizing under-resourced jurisdictions."
    )


def _render_deviation(df: pd.DataFrame, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Deviation from Historical Baseline")
    st.caption("Values > 1.0 exceed historical norm · Values > 2.0 are notable tail signals")

    sort_hazard = st.selectbox("Hazard to map / sort by", ["Wind","Fire","Flood","Winter","Max (any hazard)"])
    sort_col    = {'Wind':'dev_wind_p','Fire':'dev_fire_p','Flood':'dev_flood_p',
                   'Winter':'dev_winter_p','Max (any hazard)':'dev_max'}[sort_hazard]
    cscale      = {'Wind':'wind','Fire':'fire','Flood':'flood',
                   'Winter':'winter','Max (any hazard)':'deviation'}[sort_hazard]

    # Sort on df before subsetting — dev_max not in display columns
    sort_key = sort_col if sort_col != "dev_max" else "dev_max"
    df_sorted = df.sort_values(sort_key, ascending=False)
    # Map uses dev_max directly on df; table uses per-hazard dev cols
    map_col = sort_col
    fig = make_choropleth(df, map_col, f"{sort_hazard} Deviation × Baseline",
                          geojson=geojson, color_scale=cscale,
                          hover_cols=['state','county', map_col, 'max_hazard'],
                          range_color=(0, df[map_col].quantile(0.98)), map_style=map_style)

    display = df_sorted[['state','county','fire_p','dev_fire_p','wind_p','dev_wind_p',
                           'flood_p','dev_flood_p','winter_p','dev_winter_p',
                           'max_hazard','multi_hazard_flag','elevated_hazards']].copy()
    for c in ['fire_p','wind_p','flood_p','winter_p']:
        display[c] = display[c].apply(lambda x: f"{x:.1%}")
    for c in ['dev_fire_p','dev_wind_p','dev_flood_p','dev_winter_p']:
        display[c] = display[c].apply(lambda x: f"{x:.2f}×")
    display['multi_hazard_flag'] = display['multi_hazard_flag'].apply(lambda x: "⚠️" if x else "")
    display.columns = ['State','County','Fire %','Fire Dev','Wind %','Wind Dev',
                       'Flood %','Flood Dev','Winter %','Winter Dev',
                       'Dominant Hazard','Multi⚠️','Elevated Hazards']

    show_map_and_table(fig, display)


def _render_multi_hazard(df: pd.DataFrame, threshold: float, geojson: dict = None, map_style: str = "Dark"):
    flagged = df[df['multi_hazard_flag']].copy()
    st.subheader("Multi-Hazard Compound Risk Flags")
    st.caption(f"Counties where **{MULTI_HAZARD_MIN_COUNT}+** hazards exceed **{threshold:.1f}×** historical baseline simultaneously.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged counties",    len(flagged))
    c2.metric("3-hazard flags",      int((flagged['multi_hazard_count'] >= 3).sum()))
    c3.metric("% of all counties",   f"{len(flagged)/len(df)*100:.1f}%")

    # Map: number of elevated hazards across all counties (not just flagged)
    fig = make_choropleth(df, 'multi_hazard_count',
                          f"# Hazards > {threshold:.1f}× Baseline",
                          geojson=geojson, color_scale='multi',
                          hover_cols=['state','county','elevated_hazards','max_hazard'],
                          range_color=(0, 4), map_style=map_style)

    sort_col = st.selectbox("Sort table by", ["# Hazards Elevated","CEL Total"])
    sort_map = {"# Hazards Elevated":"multi_hazard_count","CEL Total":"cel_total"}

    display = flagged[['state','county','multi_hazard_count','elevated_hazards',
                        'dev_fire_p','dev_flood_p','dev_wind_p','dev_winter_p',
                        'cel_total','max_hazard','Population (2020)']].copy()
    display = display.sort_values(sort_map[sort_col], ascending=False)
    for c in ['dev_fire_p','dev_flood_p','dev_wind_p','dev_winter_p']:
        display[c] = display[c].apply(lambda x: f"{x:.2f}×")
    display['cel_total']          = display['cel_total'].apply(lambda x: f"${x:,.0f}")
    display['Population (2020)']  = display['Population (2020)'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    display.columns = ['State','County','# Hazards','Elevated Hazards',
                       'Fire Dev','Flood Dev','Wind Dev','Winter Dev',
                       'CEL Total','Dominant Hazard','Population']

    show_map_and_table(fig, display)


def _render_delta(rapt: pd.DataFrame, window: int, state_filter: str, geojson: dict = None, map_style: str = "Dark"):
    st.subheader("Month-over-Month Delta: June → July")
    st.caption("Change in AHI probability between June and July forecasts.")

    for m in [6, 7]:
        if not (PREDICTIONS_DIR / f"national_predictions_month{m:02d}.csv").exists():
            st.error(f"Missing predictions for {AVAILABLE_MONTHS[m]}. Both months required.")
            st.stop()

    df6 = compute_deviations(merge_data(load_predictions(6), rapt))
    df7 = compute_deviations(merge_data(load_predictions(7), rapt))

    if state_filter and len(state_filter) == 2:
        df6 = df6[df6['state'] == state_filter]
        df7 = df7[df7['state'] == state_filter]

    delta = compute_month_delta(df6, df7)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Counties with spike > +15%", int((delta['max_spike'] >= 0.15).sum()))
    c2.metric("Counties with drop > −15%",  int((delta['max_drop'] <= -0.15).sum()))
    c3.metric("Avg wind delta",  f"{delta['delta_wind_p'].mean():+.1%}")
    c4.metric("Avg flood delta", f"{delta['delta_flood_p'].mean():+.1%}")

    map_hazard = st.selectbox("Map delta for hazard",
                              ["Wind","Fire","Flood","Winter","Max spike (any)"])
    map_col    = {'Wind':'delta_wind_p','Fire':'delta_fire_p','Flood':'delta_flood_p',
                  'Winter':'delta_winter_p','Max spike (any)':'max_spike'}[map_hazard]
    p98 = delta[map_col].abs().quantile(0.98)

    fig = make_choropleth(delta, map_col,
                          f"Δ {map_hazard} Risk (Jul − Jun)",
                          geojson=geojson, color_scale='delta',
                          hover_cols=['state','county','spike_hazard','drop_hazard'],
                          range_color=(-p98, p98), map_style=map_style)

    view = st.radio("Show in table", ["Biggest spikes","Biggest drops","All counties"], horizontal=True)
    sort_col  = {'Wind':'delta_wind_p','Fire':'delta_fire_p','Flood':'delta_flood_p',
                 'Winter':'delta_winter_p','Max spike (any)':'max_spike'}[map_hazard]
    ascending = (view == "Biggest drops")
    if view == "Biggest spikes":
        disp_df = delta[delta['max_spike'] >= 0.05]
    elif view == "Biggest drops":
        disp_df = delta[delta['max_drop'] <= -0.05]
    else:
        disp_df = delta.copy()
    disp_df = disp_df.sort_values(sort_col, ascending=ascending)

    out = disp_df[['state','county',
                   'fire_p_June','fire_p_July','delta_fire_p',
                   'flood_p_June','flood_p_July','delta_flood_p',
                   'wind_p_June','wind_p_July','delta_wind_p',
                   'winter_p_June','winter_p_July','delta_winter_p',
                   'spike_hazard','drop_hazard']].copy()
    for c in ['fire_p_June','fire_p_July','flood_p_June','flood_p_July',
              'wind_p_June','wind_p_July','winter_p_June','winter_p_July']:
        out[c] = out[c].apply(lambda x: f"{x:.1%}")
    for c in ['delta_fire_p','delta_flood_p','delta_wind_p','delta_winter_p']:
        out[c] = out[c].apply(lambda x: f"{x:+.1%}")
    out.columns = ['State','County',
                   'Fire Jun','Fire Jul','Δ Fire',
                   'Flood Jun','Flood Jul','Δ Flood',
                   'Wind Jun','Wind Jul','Δ Wind',
                   'Winter Jun','Winter Jul','Δ Winter',
                   'Spike Hazard','Drop Hazard']

    show_map_and_table(fig, out)
    st.caption("Δ = July − June. Positive = increased risk month over month.")


if __name__ == "__main__":
    main()
