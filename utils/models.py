from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def xgboost_modelling(X_train, y_train, X_test, y_test):  
    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [3, 6, 10, 15, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 10, 15],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
        'scale_pos_weight': [1, 2, 5, 10],
        'objective': ['binary:logistic', 'reg:squarederror']
    }

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, 
                                   cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    
    y_train = y_train.values.ravel()

    xgb_grid_search.fit(X_train, y_train)
    
    xgb_best_model = xgb_grid_search.best_estimator_

    xgb_pred = xgb_best_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    return xgb_accuracy