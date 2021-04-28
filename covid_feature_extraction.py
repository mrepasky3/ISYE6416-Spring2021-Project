import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trips = pd.read_csv("data/Trips_by_Distance.csv")
epicurve_report_date = pd.read_csv("data/epicurve_rpt_date.csv")

def county_extraction(county_name, target_label):
    '''
    Format the data such that the feature matrix X contains the data vectors
    parametrizing the pandemic and the label vector y contains the information
    specified by the target_label parameter for a county specified by county_name.
    
    Parameters
    ----------
    - county_name : string county name for extraction
    - target_label : string specifying either the Population, Long, Medium, or
        Short corresponding to fractional population not staying home on a given
        day and the fractions of long (>100 miles), medium (10-100 miles), and
        short (<10 miles) trips in the county on a given day.
    Returns
    -------
    - county_trip_dates : the date corresponding to each datum
    - county_y : the y vector as specified by the target_label parameter
    - feature_matrix : matrix containing numbers, totals, and moving averages
        of the cases and deaths per 100k residents
    - feature_labels : the names of the features corresponding to the columns
        of the feature matrix
    '''
    trip_indices = np.logical_and(np.array(trips["County Name"] == (county_name + " County")), np.array(trips["State Postal Code"] == "GA"))
    trip_indices = np.logical_and(trip_indices, np.array(pd.to_datetime(trips["Date"]) >= "2020-02-29"))
    trip_indices = np.logical_and(trip_indices, np.array(pd.to_datetime(trips["Date"]) <= "2021-04-10"))
    county_travel = trips.loc[trip_indices].reset_index()

    county_population = (county_travel["Population Staying at Home"] + county_travel["Population Not Staying at Home"]).iloc[0]
    county_trip_dates = pd.to_datetime(county_travel["Date"])
    
    if target_label == "Population":
        county_travelers = county_travel["Population Not Staying at Home"]
        county_y = county_travelers/county_population
    elif target_label == "Long":
        county_long_trips = county_travel["Number of Trips 100-250"] + county_travel["Number of Trips 250-500"] + county_travel["Number of Trips >=500"]
        county_y = county_long_trips/county_travel["Number of Trips"]
    elif target_label == "Medium":
        county_med_trips = county_travel["Number of Trips 10-25"] + county_travel["Number of Trips 25-50"] + county_travel["Number of Trips 50-100"]
        county_y = county_med_trips/county_travel["Number of Trips"]
    elif target_label == "Short":
        county_short_trips = county_travel["Number of Trips <1"] + county_travel["Number of Trips 1-3"] + county_travel["Number of Trips 3-5"] + county_travel["Number of Trips 5-10"]
        county_y = county_short_trips/county_travel["Number of Trips"]
    
    case_indices = np.logical_and(np.array(epicurve_report_date["county"] == county_name), np.array(epicurve_report_date["report_date"] <= "2021-04-10"))
    case_indices = np.logical_and(case_indices, np.array(epicurve_report_date["report_date"] >= "2020-02-29"))
    county_cases = epicurve_report_date.loc[case_indices].reset_index()
    
    county_case_dates = pd.to_datetime(county_cases["report_date"])
    county_case_numbers = county_cases["total_cases"]
    county_case_cum = county_cases["total_cases_cum"]
    county_case_ma = county_cases["moving_avg_total_cases"]
    county_death_numbers = county_cases["deaths"]
    county_death_cum = county_cases["death_cum"]
    county_death_ma = county_cases["moving_avg_deaths"]
    
    county_case_frac = 1e5*county_case_numbers/county_population
    county_death_frac = 1e5*county_death_numbers/county_population
    
    county_case_cum_frac = 1e5*county_case_cum/county_population
    county_death_cum_frac = 1e5*county_death_cum/county_population
    
    county_case_ma_frac = 1e5*county_case_ma/county_population
    county_death_ma_frac = 1e5*county_death_ma/county_population

    feature_labels = ["Cases per 100k", "Cumulative Cases per 100k", "Moving Average Cases per 100k",
                      "Deaths per 100k", "Cumulative Deaths per 100k", "Moving Average Deaths per 100k"]
    feature_matrix = pd.concat([county_case_frac, county_case_cum_frac, county_case_ma_frac,
                                county_death_frac, county_death_cum_frac, county_death_ma_frac],axis=1)
    return county_trip_dates, county_y, feature_matrix, feature_labels


def state_extraction(target_label):
    '''
    Format the data such that the feature matrix X contains the data vectors
    parametrizing the pandemic and the label vector y contains the information
    specified by the target_label parameter for the state of Georgia.
    
    Parameters
    ----------
    - target_label : string specifying either the Population, Long, Medium, or
        Short corresponding to fractional population not staying home on a given
        day and the fractions of long (>100 miles), medium (10-100 miles), and
        short (<10 miles) trips in Georgia on a given day.
    Returns
    -------
    - state_trip_dates : the date corresponding to each datum
    - state_y : the y vector as specified by the target_label parameter
    - state_X : matrix containing numbers, totals, and moving averages
        of the cases and deaths per 100k residents
    - feature_labels : the names of the features corresponding to the columns
        of the feature matrix
    '''
    trip_indices = np.array(trips["State Postal Code"] == "GA")
    trip_indices = np.logical_and(trip_indices, np.array(pd.to_datetime(trips["Date"]) >= "2020-02-29"))
    trip_indices = np.logical_and(trip_indices, np.array(pd.to_datetime(trips["Date"]) <= "2021-04-10"))
    state_travel = trips.loc[trip_indices].reset_index()
    trip_indices = np.where(pd.isnull(state_travel["County Name"]))
    state_travel = state_travel.loc[trip_indices]

    state_population = (state_travel["Population Staying at Home"] + state_travel["Population Not Staying at Home"]).iloc[0]
    state_trip_dates = pd.to_datetime(state_travel["Date"])

    if target_label == "Population":
        state_travelers = state_travel["Population Not Staying at Home"]
        state_y = state_travelers/state_population
    elif target_label == "Long":
        state_long_trips = state_travel["Number of Trips 100-250"] + state_travel["Number of Trips 250-500"] + state_travel["Number of Trips >=500"]
        state_y = state_long_trips/state_travel["Number of Trips"]
    elif target_label == "Medium":
        state_med_trips = state_travel["Number of Trips 10-25"] + state_travel["Number of Trips 25-50"] + state_travel["Number of Trips 50-100"]
        state_y = state_med_trips/state_travel["Number of Trips"]
    elif target_label == "Short":
        state_short_trips = state_travel["Number of Trips <1"] + state_travel["Number of Trips 1-3"] + state_travel["Number of Trips 3-5"] + state_travel["Number of Trips 5-10"]
        state_y = state_short_trips/state_travel["Number of Trips"]

    case_indices = np.logical_and(np.array(epicurve_report_date["county"] == "Georgia"), np.array(epicurve_report_date["report_date"] <= "2021-04-10"))
    case_indices = np.logical_and(case_indices, np.array(epicurve_report_date["report_date"] >= "2020-02-29"))
    state_cases = epicurve_report_date.loc[case_indices].reset_index()

    state_case_dates = pd.to_datetime(state_cases["report_date"])
    state_case_numbers = state_cases["total_cases"]
    state_case_cum = state_cases["total_cases_cum"]
    state_case_ma = state_cases["moving_avg_total_cases"]
    state_death_numbers = state_cases["deaths"]
    state_death_cum = state_cases["death_cum"]
    state_death_ma = state_cases["moving_avg_deaths"]

    state_case_frac = 1e5*state_case_numbers/state_population
    state_death_frac = 1e5*state_death_numbers/state_population

    state_case_cum_frac = 1e5*state_case_cum/state_population
    state_death_cum_frac = 1e5*state_death_cum/state_population

    state_case_ma_frac = 1e5*state_case_ma/state_population
    state_death_ma_frac = 1e5*state_death_ma/state_population

    feature_labels = ["Cases per 100k", "Cumulative Cases per 100k", "Moving Average Cases per 100k",
                      "Deaths per 100k", "Cumulative Deaths per 100k", "Moving Average Deaths per 100k"]
    state_X = pd.concat([state_case_frac, state_case_cum_frac, state_case_ma_frac,
                         state_death_frac, state_death_cum_frac, state_death_ma_frac],axis=1)
    return state_trip_dates, state_y, state_X, feature_labels