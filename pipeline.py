from clearml import PipelineDecorator, Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


@PipelineDecorator.component(cache=True, execution_queue="default")
def load_data(url: str):
    df = pd.read_csv(url)
    if df.isnull().values.any():
        print("Данные содержат NaN значения.")
        df.dropna(inplace=True)
    return df


@PipelineDecorator.component(cache=True, execution_queue="default")
def train_model(X, y, model_params):
    model = RandomForestClassifier(**model_params)
    model.fit(X, y)
    return model


@PipelineDecorator.component(cache=True, execution_queue="default")
def log_results(task: Task, model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    task.get_logger().report_text(report)


@PipelineDecorator.pipeline(
    name='Credit Default Prediction Pipeline',
    project='Credit Default Prediction',
    version='0.1'
)


def pipeline_logic(url: str):
    task = Task.init(project_name='Credit Default Prediction', task_name='Random Forest Experiment', task_type=Task.TaskTypes.optimizer)
    
    # Загрузка данных
    df = load_data(url)
    X = df[['Income', 'Age', 'Loan', 'Loan to Income']]
    y = df['Default']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Определение параметров модели
    model_params_1 = {
        'n_estimators': 100,
        'max_depth': 1,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'bootstrap': True,
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    # Обучение первой модели
    model_1 = train_model(X_train, y_train, model_params_1)
    log_results(task, model_1, X_test, y_test)

    # Определение параметров второй модели
    model_params_2 = {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'bootstrap': False,
        'class_weight': None,
        'random_state': 42
    }
    
    # Обучение второй модели
    model_2 = train_model(X_train, y_train, model_params_2)
    log_results(task, model_2, X_test, y_test)

    task.close()


if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=1EwBF6y6DIZvacQ56PPHVxZilFR_Mk_dN'
    PipelineDecorator.run_locally()  # Для локального запуска
    pipeline_logic(url)
  
