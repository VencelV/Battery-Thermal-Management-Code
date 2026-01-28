import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df_raw = pd.read_csv('temp_values.csv')#goes through the Simulink data table
temp_breakpoints = [0, 10, 25, 45]#corresponding temperature breakpoints
parsed_data = []#where the data is stored

for index, row in df_raw.iterrows():
    soc = row['state of charge '] # I accidentally added a space in the column name in the CSV oopsies :)
    r0_values = [float(x) for x in row['series resistance table data'].strip().split()]# converts the strings into a list
    ocv_values = [float(x) for x in row['open circuit voltage table data'].strip().split()]
    for i, temp in enumerate(temp_breakpoints):#match values to their temperature 
        parsed_data.append({
            'Temperature': temp,
            'SOC': soc,
            'R0': r0_values[i],  # Series Resistance
            'OCV': ocv_values[i] # Open Circuit Voltage
        })
data = pd.DataFrame(parsed_data)#makes a clean dataframe
X = data[['R0', 'OCV', 'SOC']]# separate features (X) and target variable (y)
y = data['Temperature']# we use R0 and OCV because they are the only parameters in the simulink table data (very weird)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# split data into training and testing sets

model = RandomForestClassifier(n_estimators=100, random_state=42)#selects the model

model.fit(X_train, y_train)#trains it

y_pred = model.predict(X_test)#model evaluates the given
accuracy = accuracy_score(y_test, y_pred)

print("Parsed Data Sample:")
print(data.head())
print("-" * 30)
print(f"Model Accuracy: {accuracy}")
print(f"Predicted Temperatures: {y_pred}")
print(f"Actual Temperatures:    {y_test.values}")