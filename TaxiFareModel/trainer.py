# imports
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import test_train
from TaxiFareModel.data import clean_data
from TaxiFareModel.data import get_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_X_y




class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    # implement set_pipeline() function
    # defines the pipeline as a class attribute
    def set_pipeline(self):

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
         ('preproc', preproc_pipe),
         ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return pipe


    def test_train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        pass

    #set and train the pipeline
    def run(self):
        self.pipeline.fit(self.X_train, self.y_train)
        pass

    # implement evaluate() function
    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse



if __name__ == "__main__":
    df = get_data(nrows=10_000)
    df = clean_data(df)
    X,y = get_X_y(df)
    trainer = Trainer(X,y)
    trainer.set_pipeline()
    trainer.test_train()
    trainer.run()
    trainer.evaluate()

    print('TODO')
