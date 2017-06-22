import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    try:
        data_df = pd.read_csv("auto.csv")
    except Exception as e:
        print(e)
        sys.exit(1)

    print(data_df.head())

    # what are the regions?
    unique_regions = data_df["origin"].unique()
    print(unique_regions)

    # *** convert discrete values to categorical variables

    # create binary values for each of the 'cylinders' values. the columns will be prefixed by 'cyl'
    cylinders_df = pd.get_dummies(data_df["cylinders"], prefix="cyl")
    data_df = pd.concat([data_df, cylinders_df], axis=1)

    # create binary values for each of the 'years' values. the columns will be prefixed by 'year'
    year_df = pd.get_dummies(data_df["year"], prefix="year")
    data_df = pd.concat([data_df, year_df], axis=1)

    data_df = data_df.drop("cylinders", axis=1)
    data_df = data_df.drop("year", axis=1)

    print(data_df.head())

    # split the data into train and test data frames after shuffling
    shuffled_rows = np.random.permutation(data_df.index)
    shuffled_cars = data_df.iloc[shuffled_rows]
    highest_train_row = int(data_df.shape[0] * .70)
    train = shuffled_cars.iloc[0:highest_train_row]
    test = shuffled_cars.iloc[highest_train_row:]

    print(len(train))
    print(len(test))

    features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]
    print(features)

    models = {}
    # because we are doing one-versus-all method. we will fit unique_regions amount of models
    for origin in unique_regions:
        model = LogisticRegression()

        X_train = train[features]
        y_train = train["origin"] == origin # this is what separates the models based on origin

        model.fit(X_train, y_train)
        models[origin] = model
        accuracy = model.score(X_train, y_train)
        print("Accuracy Score for model of origin {} is {}".format(origin, accuracy))

    # predict probabilities
    testing_probs = pd.DataFrame(columns=unique_regions) # this data frame will contain the prob by region
    for origin in unique_regions:
        X_test = test[features]
        # probability of observation being in the origin
        testing_probs[origin] = models[origin].predict_proba(X_test)[:, 1]

    print(testing_probs.head())

    # return a Series where each value corresponds to the column or where the maximum value occurs for that observation
    predicted_origins = testing_probs.idxmax(axis=1)
    print(predicted_origins)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)