from clearml import PipelineDecorator, Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@PipelineDecorator.component(cache=True, execution_queue="default")
def load_data(url: str):    
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("Загруженный DataFrame пуст.")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки

    # Проверка наличия необходимых столбцов
    required_columns = ['Income', 'Age', 'Loan', 'Loan to Income', 'Default']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует необходимый столбец: {col}")

    if df.isnull().values.any():
        print("Данные содержат NaN значения.")
        print(f"Удалено {df.isnull().sum().sum()} строк с NaN значениями.")
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
    report = classification_report(y_test, y_pred, output_dict=True)
    task.get_logger().report_text(str(report))
    
    # Log accuracy
    accuracy = accuracy_score(y_test, y_pred)
    task.get_logger().report_scalar("Accuracy", "Model Accuracy", value=accuracy)
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    task.get_logger().report_matplotlib_figure("Confusion Matrix", "Confusion Matrix", plt)
    plt.close()
    
    # Log ROC AUC score
    if len(set(y_test)) == 2:  # Check if binary classification
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        task.get_logger().report_scalar("ROC AUC", "ROC AUC Score", value=roc_auc)

@PipelineDecorator.pipeline(
    name='Credit Default Prediction Pipeline',
    project='Credit Default Prediction',
    version='0.1'
)
def pipeline_logic(url: str):
    current_task = Task.current_task()
    if current_task:
        current_task.close()
    
    task = Task.init(project_name='Credit Default Prediction',
                     task_name='Random Forest Experiment',
                     task_type=Task.TaskTypes.optimizer)
    
    df = load_data(url)    
    if df.empty:
        raise ValueError("Не удалось загрузить данные. Проверьте URL или формат данных.")
    
    X = df[['Income', 'Age', 'Loan', 'Loan to Income']]
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
    model_1 = train_model(X_train, y_train, model_params_1)
    log_results(task, model_1, X_test, y_test)

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
    
    model_2 = train_model(X_train, y_train, model_params_2)
    log_results(task, model_2, X_test, y_test)

    task.close()

if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=1EwBF6y6DIZvacQ56PPHVxZilFR_Mk_dN'
    PipelineDecorator.run_locally()
    pipeline_logic(url) 
