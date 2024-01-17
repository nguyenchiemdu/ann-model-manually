import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from lib.ann import ANN
from lib.layer import Layer
from sklearn.metrics import mean_squared_error, r2_score

# Import data
data = pd.read_csv("./AKH_WQI.csv")
y = data["WQI"]
X = data.drop(columns=["WQI"])

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define model
model = ANN()
model.addLayer(Layer(X.columns.__len__(), 128, "relu", 1))
model.addLayer(Layer(128, 64, "linear", 2))
model.addLayer(Layer(64, 64, "relu", 3))
model.addLayer(Layer(64, 32, "linear", 4))
model.addLayer(Layer(32, 32, "relu", 5))
model.addLayer(Layer(32, 1, "linear", 6))

# Train model
model.train(X_train, y_train, epoch=1000, cal_mse=True)
y_pred = model.predict(X_test)

# Evaludate model
for i in range(X_test.shape[0]):
    print(
        "Predicted: ",
        (model.predict(X_test.iloc[i])).squeeze(),
        "Target: ",
        y_test.iloc[i],
    )

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
