from clearml import Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def main():
    # Инициализация задачи
    task = Task.init(project_name='Credit Default Prediction',
                    task_name='Random Forest Experiment',
                    task_type=Task.TaskTypes.optimizer)

    # Загрузка данных
    url = 'https://drive.google.com/uc?id=1EwBF6y6DIZvacQ56PPHVxZilFR_Mk_dN'
    df = pd.read_csv(url)

    # Проверка на NaN в данных
    if df.isnull().values.any():
        print("Данные содержат NaN значения.")
        df.dropna(inplace=True)

    X = df[['Income', 'Age', 'Loan', 'Loan to Income']]    
    y = df['Default']

    # Разделение исходных данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Определение и обучение первой модели
    model_1 = RandomForestClassifier(
        n_estimators=100,
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='log2',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    model_1.fit(X_train, y_train)

    # Логирование метрик первой модели
    y_pred_1 = model_1.predict(X_test)
    task.get_logger().report_text("Model 1 Classification Report:\n" + classification_report(y_test, y_pred_1))

    # Определение и обучение второй модели
    model_2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=False,
        class_weight=None,
        random_state=42
    )
    model_2.fit(X_train, y_train)

    # Логирование метрик второй модели
    y_pred_2 = model_2.predict(X_test)
    task.get_logger().report_text("Model 2 Classification Report:\n" + classification_report(y_test, y_pred_2))

    # Завершение задачи
    task.close()


if __name__ == "__main__":
    main()