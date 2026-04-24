from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)
