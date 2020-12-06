import pandas as pd
from sklearn.model_selection import train_test_split

def transform_ordinal_cat(value, mydict):

    return mydict[value]

def condition_test(test, color_dict, clarity_dict, cut_dict):

    test = test.drop(['depth', 'id'], axis = 1)
    test['color_val'] = test.color.apply(lambda x: transform_ordinal_cat(x,color_dict ))
    test['clar_val'] = test.clarity.apply(lambda x: transform_ordinal_cat(x,clarity_dict ))
    test['cut_val'] = test.cut.apply(lambda x: transform_ordinal_cat(x,cut_dict))
    test = test.drop(['cut','color','clarity'], axis = 1)



    return test


def create_csv(ids, values, filepath):

    submit = pd.DataFrame({'id': ids,'price':values})

    submit.to_csv(filepath, index= False)


def prepare_data(df, color_dict, clarity_dict, cut_dict, test_size):

    df = df.drop(['depth', 'id'], axis = 1)

    df['color_val'] = df.color.apply(lambda x: transform_ordinal_cat(x,color_dict ))
    df['clar_val'] = df.clarity.apply(lambda x: transform_ordinal_cat(x,clarity_dict ))
    df['cut_val'] = df.cut.apply(lambda x: transform_ordinal_cat(x,cut_dict))
    X = df.drop(['price','cut','color','clarity'], axis = 1)
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=test_size, 
                                                    random_state=123)

    return X_train, X_test, y_train, y_test