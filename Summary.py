from __future__ import division
from cx_Oracle import connect
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from StringIO import StringIO
from copy import deepcopy
from datetime import datetime

from IPython.core.display import display, Image
from cx_Oracle import DatabaseError
from pydotplus import graph_from_dot_data

from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sqlalchemy import create_engine
from sqlalchemy.dialects.oracle.base import VARCHAR2


def my_create_alchemy_conn(user, pw, db='PXXDBM1'):
    eng = create_engine('oracle+cx_oracle://' + user + ':' + pw + '@' + db, label_length=29)
    connection = eng.connect()
    return eng, connection


def my_write_sql(df, table_name, user, pw, db='PXXDBM1', replace_old = False, varchar_length = 200):
    print 'Exporting to {0} ({1})'.format(table_name, datetime.now())
    eng, conn = my_create_alchemy_conn(user, pw)

    # Replace old table
    if replace_old:
        try:
            conn.execute("DROP TABLE {0}".format(table_name))
        except:
            print "No table to replace"

    # Make all variables to_upper
    df.columns = [col.upper() for col in df.columns]

    # Make correct dtype for all object variables (otherwise Objects which doesn't work in Oracle)
    cols = df.dtypes[df.dtypes == object].index
    type_mapping = {col: VARCHAR2(varchar_length) for col in cols}
    df.to_sql(table_name, con=eng, index=False, chunksize=10000, dtype=type_mapping)
    conn.close()
    print 'Finished ({0})'.format(datetime.now())
    return df[0:3]


def my_sql_request(q, user, pw, displayed=True, db='PXXDBM1'):
    """Connects to SQL, executes query, and returns a dataframe representing SQL table gained from query, also prints
    it if you want

    :param q: Your query
    :param user: UserName
    :param pw: Password
    :param db: Database used
    :param displayed: Display table?
    :return: Dataframe representing the table from the query
    """
    df = True
    try:
        connection = connect(user, pw, db)
        cursor = connection.cursor()
        cursor.execute(q)

        if ('create table' in q.lower() or 'create view' in q.lower()) and displayed:
            q = q.split('AS', 1)[1]
            df = my_sql_request(q, user, pw, displayed)

        elif 'drop' in q.lower():
            pass

        else:
            df = pd.DataFrame(cursor.fetchall())
            df.columns = [i[0] for i in cursor.description]
            if displayed:
                display(df[0:3])
                my_missing_value_checker(df)

        cursor.close()
    except (DatabaseError, TypeError) as e:
        raise DatabaseError('{0}, {1}'.format(e, q))
    return df


def my_tree_plot(df, x_train, y_train, x_test, y_test, max_depth, class_names=None, class_weight='balanced'):
    """Create a plot of a decision tree
    :param df:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param max_depth:
    :param class_names:
    :return:
    """
    print 'Using my_tree_plot ({0})'.format(datetime.now())

    dot_data = StringIO()
    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
    clf = clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    no_right = sum(preds == y_test) * 1.0
    print 'Accuracy {0}'.format(no_right / len(preds))
    export_graphviz(clf, out_file=dot_data, feature_names=df.columns, filled=True, rounded=True,
                    special_characters=True, proportion=True, class_names=class_names)
    graph = graph_from_dot_data(dot_data.getvalue())

    display(Image(graph.create_png()))
    return graph


def my_missing_value_checker(df):
    """Checks if values are missing

    :param df: A dataframe with possibly missing values
    :type df: Pandas.DataFrame
    :return: A Series with all variables with na_values >= 1
    :rtype: pandas.Series
    """
    print 'Using my_missing_value_checker ({0})'.format(datetime.now())
    nas = df.isnull().sum()
    print "Missing values"
    print nas[nas > 0]
    return nas[nas > 0]


def my_drop_wrong_type(df):
    """
    Drops all columns that are not in column list (int, float, np.int64, np.float64) and informs user about drop

    :param df: DataFrame with incorrect columns to drop
    :type df: Pandas.DataFrame
    :rtype: Pandas.DataFrame
    """
    print 'Using my_drop_wrong_type ({0})'.format(datetime.now())

    type_list = [int, float, np.int64, np.float64]
    drop_list = []
    for col in df:
        if type(df.loc[0, col]) not in type_list:
            drop_list.append(col)

    df2 = deepcopy(df).drop(drop_list, 1, inplace=False)
    print 'Dropped: {0}'.format(drop_list)
    return df2


def my_visual_parameter_tuning(x_train, y_train, pipe, param_name, n_fold, param_range):
    """Shows a plot to see which value for a parameter is the best.
    
    Args:
        x_train (DataSeries): X variable
        y_train (DataSeries): y variable
        pipe (Pipeline): Pipeline with the input set
        n_fold (int): How often the cross validation is done
        param_name (String): The name of the parameter that should be tested
        param_range (List): List of a range of parameter values

    Returns:
        bool: True if the function worked
    """
    print 'Using my_visual_parameter_tuning ({0})'.format(datetime.now())
    # Visualization Check
    train_scores, test_scores = validation_curve(estimator=pipe, X=x_train, y=y_train, param_name=param_name,
                                                 param_range=param_range, cv=n_fold, n_jobs=1, verbose=1)
    # Get mean and SD
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot Training and test group accuracy
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')                
    plt.grid()
    plt.legend(loc='best')        
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.show()
    return True


def my_roc_curve(true, pred):
    """Compute a ROC-curve based on predicted and true values
    
    Args:
        true (DataSeries): true values
        pred (DataSeries): predicted values
    """
    print 'Using my_roc_curve ({0})'.format(datetime.now())

    fpr, tpr, _ = roc_curve(true, pred)
    print('Creating ROC not possible')
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    return True


def my_learning_curve(x_train, y_train, pipe, n_fold,  scoring):
    """Show a plot with the learning curve
    
    Args:
        x_train (DataSeries): X variable
        y_train (DataSeries): y variable
        pipe (Pipeline): Pipeline with the input set
        n_fold (int): How often cross validation is done
        scoring (String): Scoring method
    """
    print 'Using my_learning_curve ({0})'.format(datetime.now())

    # Check for over- or underfitting    
    train_sizes, train_scores, test_scores = learning_curve(scoring=scoring, estimator=pipe, X=x_train, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=n_fold, n_jobs=1,
                                                            verbose=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    return True


def my_grid_search(x_train, y_train, x_test, y_test, pipe, param_grid, scoring, n_fold, dummy_scores = None):
    """
    Do a grid search

    Args:
        :param dummy_scores: Comparison scores of the dummy variable
        :param y_test: 
        :param x_test: 
        :param x_train: X-variables
        :param y_train: y variable
        :param pipe: the pipeline used
        :param param_grid: The parameters used for the model
        :param n_fold: n number of CV-tries
        :param scoring: A string (see model evaluation documentation) or a scorer callable object / function with signature
                    scorer (estimator, X, y). If None, the score method of the estimator is used.

    Returns:
        gs.best_estimator_: Best estimator model
        Dict(String, Int): The scores for different models
    """
    print 'Using my_grid_search ({0})'.format(datetime.now())

    # Train
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=n_fold, n_jobs=1, verbose=1)
    gs = gs.fit(x_train, y_train)
    print '\nGridSearchCV returns:'
    print 'Best {0} Crossvalidation Score: {1}'.format(scoring, abs(gs.best_score_))
    print 'Best Parameters: {0}'.format(gs.best_params_)
    print 'Best Estimator: {0}\n'.format(gs.best_estimator_)

    # Summarize
    scores = {}
    print 'Final test with Testset'
    y_test_pred = gs.predict(x_test)
    y_test_pred_proba = gs.predict_proba(x_test)
    for score in [accuracy_score, precision_score, recall_score, roc_auc_score]:
        try:
            test_score = score(y_test, y_test_pred)
        except ValueError as e:
            print('Creating score not possible - {0}'.format(e))
        else:
            print 'Testset {1}: {0}'.format(abs(test_score), score.__name__)
            scores[score.__name__] = test_score
            # If dummy variable was given, create a comparison of test_score / dummy_score
            if score.__name__ in dummy_scores.keys():
                print '{0} test/dummy: {1}\n'.format(score.__name__, abs(test_score / dummy_scores[score.__name__]))

    # Plot
    print '\nTestset: my_roc_curve'
    my_roc_curve(y_test, y_test_pred)
    print 'true negative, false negative'
    print 'false positive, true positive'
    print confusion_matrix(y_test_pred, y_test)
    print '\nTestset: (length of top 10%, lift): {0}'.format(my_lift(y_test, y_test_pred_proba))
    print '\nAll data: my_learning_curve'
    my_learning_curve(pd.concat([x_train, x_test]), pd.concat([y_train, y_test]), pipe, n_fold, scoring)

    # Return
    return scores, gs.best_estimator_


def my_lift(y_test, y_test_pred_proba, top=0.1):
    y_test_pred = [tup[1] for tup in y_test_pred_proba]
    lift_table = pd.DataFrame(data={'orig': y_test, 'pred': y_test_pred}).sort_values(
        'pred', ascending=False)
    normal = len(lift_table[lift_table.orig == 1]) / len(lift_table)
    top_1 = lift_table[:int(top * len(lift_table))]
    top_lift = len(top_1[lift_table.orig == 1]) / len(top_1)

    try:
        lift_ratio = top_lift/normal
    except ZeroDivisionError:
        lift_ratio = 1

    return len(top_1), lift_ratio
