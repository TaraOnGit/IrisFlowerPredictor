import numpy as np
import pandas as pd

class Predict:

    def predict_Iris(self, sl, sw, pl, pw):

        df = pd.read_csv('Iris.csv')

        df.drop(columns=['Id'],inplace=True)

        X = df.drop(columns=['Species'])
        y = df['Species']

        test = np.array([float(sl),float(sw),float(pl),float(pw)])

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=2)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train,y_train)

        X_test = test.reshape(1,-1)
        y_pred = knn.predict(X_test)
        return 'This flower possibly belongs to ', str(y_pred),' specie'

#p = Predict()
#print(p.predict_Iris(5.0,2.0,4.2,0.4))
