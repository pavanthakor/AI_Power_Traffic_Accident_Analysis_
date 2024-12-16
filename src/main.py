import numpy as np
import matplotlib.pyplot as plt



def load_data(filepath):
   
    data = np.genfromtxt(filepath, delimiter=',', dtype=str, skip_header=1)
    return data

def preprocess_data(data):
   
    locations = data[:, 2]  
    dates = data[:, 0]      

    try:
        severities = data[:, 3].astype(int)
    except ValueError:
       
        severities = np.array([int(val) if val.isdigit() else 0 for val in data[:, 3]])

    return locations, severities, dates


def high_risk_locations(locations, severities):
   
    unique_locations = np.unique(locations)
    location_risks = {loc: np.sum(severities[locations == loc]) for loc in unique_locations}
    sorted_locations = sorted(location_risks.items(), key=lambda x: x[1], reverse=True)
    return sorted_locations

def plot_high_risk_locations(location_data):
  
    locations, risks = zip(*location_data)
    plt.bar(locations, risks, color='red')
    plt.title("High-Risk Locations by Severity")
    plt.xlabel("Locations")
    plt.ylabel("Total Severity")
    plt.savefig("visuals/high_risk_locations.png")
    plt.show()


def plot_severity_over_time(dates, severities):
  
    unique_dates = np.unique(dates)
    severity_per_date = [np.sum(severities[dates == date]) for date in unique_dates]
    
    plt.plot(unique_dates, severity_per_date, marker='o', color='blue')
    plt.title("Severity of Accidents Over Time")
    plt.xlabel("Dates")
    plt.ylabel("Total Severity")
    plt.xticks(rotation=45)
    plt.savefig("visuals/severity_over_time.png")
    plt.show()


if __name__ == "__main__":

    data = load_data("data/accident_data.csv")
    locations, severities, dates = preprocess_data(data)

    
    location_data = high_risk_locations(locations, severities)
    print("High-Risk Locations:", location_data)

    plot_high_risk_locations(location_data)
    plot_severity_over_time(dates, severities)
