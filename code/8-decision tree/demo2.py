X = housing_boston.iloc[:, :5].values
Y = housing_boston.iloc[:, 6].values
oe = OneHotEncoder()
X = oe.fit_transform(X).toarray()
label_Y = LabelEncoder()
Y = label_Y.fit_transform(Y)