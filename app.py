import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np

def find_nearest_core_point(dbscan, X_scaled, new_data_scaled):
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    core_points = X_scaled[core_samples_mask]
    distances = np.linalg.norm(core_points - new_data_scaled, axis=1)
    nearest_core_point_index = np.argmin(distances)
    return dbscan.labels_[dbscan.core_sample_indices_[nearest_core_point_index]]

def task1_dbscan():
    st.header("Task 1: DBSCAN Clustering")
    data_task1 = pd.read_csv("train.xlsx - train.csv")
    X = data_task1.drop("target", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    optimal_params = {"eps": 1.1999999999999997, "min_samples": 5}
    eps_optimal = optimal_params["eps"]
    min_samples_optimal = optimal_params["min_samples"]
    dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_optimal)
    dbscan.fit(X_scaled)
    data_task1["Cluster"] = dbscan.labels_
    st.write(data_task1)

    st.subheader('Predict Cluster for a New Data Point')
    new_data = st.text_input('Enter new data point values (comma-separated):')
    if new_data:
        new_data = np.array(new_data.split(',')).astype(float).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)
        
        X_combined_scaled = np.vstack([X_scaled, new_data_scaled])
 
        combined_labels = dbscan.fit_predict(X_combined_scaled)
        
        new_data_cluster = combined_labels[-1]
        
        if new_data_cluster == -1:
            st.write("The new data point is an outlier.")
        else:
            st.write(f"The new data point belongs to cluster: {new_data_cluster}")

def task2_lightgbm():
    st.header("Task 2: LightGBM Classification")
    data_task2 = pd.read_csv("train.xlsx - train.csv")
    X = data_task2.drop("target", axis=1)
    y = pd.Categorical(data_task2["target"])
    y_codes = y.codes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codes, test_size=0.2, random_state=42
    )
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        "objective": "multiclass",
        "num_class": len(set(y_codes)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }
    lgbm_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=lgb_eval,
    )
    y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
    y_pred = [np.argmax(line) for line in y_pred]
    lgbm_acc = accuracy_score(y_test, y_pred)
    st.write(f"LightGBM Accuracy: {lgbm_acc}")

    test_data_task2 = pd.read_csv("test.xlsx - test.csv")
    test_predictions_lgbm = lgbm_model.predict(test_data_task2)
    test_predictions_lgbm = [np.argmax(line) for line in test_predictions_lgbm]
    predicted_labels = y.categories[test_predictions_lgbm]
    st.write(pd.DataFrame({"LightGBM": predicted_labels}))


def task3_aggregation():
    st.header("Task 3: Aggregation of Durations and Activities")
    raw_data = pd.read_csv("rawdata.xlsx - inputsheet.csv")
    raw_data['datetime'] = raw_data['date'] + ' ' + raw_data['time'].str[:-3]
    raw_data['datetime'] = pd.to_datetime(raw_data['datetime'], format='%m/%d/%Y %H:%M:%S')
    raw_data.sort_values(by=["location", "datetime"], inplace=True)
    raw_data["duration"] = (
        raw_data.groupby("location")["datetime"]
        .diff()
        .fillna(pd.Timedelta(seconds=0))
        .dt.total_seconds()
    )
    inside_duration = (
        raw_data[raw_data["position"].str.lower() == "inside"]
        .groupby("date")["duration"]
        .sum()
        .reset_index(name="inside_duration")
    )
    outside_duration = (
        raw_data[raw_data["position"].str.lower() == "outside"]
        .groupby("date")["duration"]
        .sum()
        .reset_index(name="outside_duration")
    )
    pick_activities = (
        raw_data[raw_data["activity"].str.lower() == "picked"]
        .groupby("date")
        .size()
        .reset_index(name="pick_activities")
    )
    place_activities = (
        raw_data[raw_data["activity"].str.lower() == "placed"]
        .groupby("date")
        .size()
        .reset_index(name="place_activities")
    )
    result = pd.merge(inside_duration, outside_duration, on="date", how="outer")
    result = pd.merge(result, pick_activities, on="date", how="outer")
    result = pd.merge(result, place_activities, on="date", how="outer")
    result.fillna(0, inplace=True)
    result["date"] = pd.to_datetime(result["date"])
    result["pick_activities"] = result["pick_activities"].astype(int)
    result["place_activities"] = result["place_activities"].astype(int)
    result["inside_duration"] = result["inside_duration"].astype(float)
    result["outside_duration"] = result["outside_duration"].astype(float)
    st.write(result)

def main():
    st.title("Machine Learning and Data Analysis Tasks")
    task = st.sidebar.selectbox(
        "Select Task",
        ["Task 1: Clustering", "Task 2: Classification", "Task 3: Aggregation"],
    )
    if task == "Task 1: Clustering":
        task1_dbscan()
    elif task == "Task 2: Classification":
        task2_lightgbm()
    elif task == "Task 3: Aggregation":
        task3_aggregation()


if __name__ == "__main__":
    main()
