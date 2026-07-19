import data_preprocessing as data_prep
from eda import run_eda

df = data_prep.fetch_data()
# run_eda(df)

X_train, X_test, y_train, y_test, preprocessor = data_prep.preprocess_data(df)
data_prep.check_class_distribution(df['is_canceled'], y_train, y_test)