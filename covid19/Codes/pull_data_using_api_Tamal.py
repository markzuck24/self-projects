
import requests
import pandas as pd

def get_data_from_api(url):
  
    print("\n")
    print(url)
    req_json = requests.get(url).json()
    data_dict = {k: pd.DataFrame(v) for k, v in req_json.items()}
    print(data_dict.keys())
    return data_dict

# Data - National time series, statewise stats and test counts
url_data = "https://api.covid19india.org/data.json"
d_data = get_data_from_api(url_data)

# Data - State-district-wise
url_state_district_wise = "https://api.covid19india.org/state_district_wise.json"
d_state_district_wise = get_data_from_api(url_state_district_wise)

# Data - Travel history
url_travel_history = "https://api.covid19india.org/travel_history.json"
d_travel_history = get_data_from_api(url_travel_history)

# Data - Raw Data
url_raw_data = "https://api.covid19india.org/raw_data.json"
d_raw_data = get_data_from_api(url_travel_history)

# Data - States Daily changes
url_states_daily = "https://api.covid19india.org/states_daily.json"
d_states_daily = get_data_from_api(url_states_daily)

