import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from typing import Dict


def fit_round(server_round: int) -> Dict:  # kết quả fit tổng hợp tính bằng trung bình có trọng số 
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: LogisticRegression): #hàm lấy loss của trung bình trọng số
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config): 
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"Log accuracy": accuracy}
    return evaluate 

def get_evaluate_fn1(model: LinearSVC): #hàm lấy loss của trung bình trọng số
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()
    
    # The `evaluate` function will be called after every round
    def evaluate1(server_round, parameters: fl.common.NDArrays, config): 
        # Update model with the latest parameters
        utils.set_model_params1(model, parameters)
        loss = log_loss(y_test, model.predict(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"SVC accuracy": accuracy}
    return evaluate1



# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    
    model1 = LinearSVC()
    utils.set_initial_params1(model1)
    strategy = fl.server.strategy.FedAvg( #chiến lược triểm khai lớp cơ sở trừu tượng 
        min_available_clients=2, 
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round, #định cấu hình đào tạo 
        
    )
    
    strategy1 = fl.server.strategy.FedAvg( #chiến lược triểm khai lớp cơ sở trừu tượng 
        min_available_clients=2, 
        evaluate_fn=get_evaluate_fn1(model1),
        on_fit_config_fn=fit_round, #định cấu hình đào tạo 
    )
    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy1,
        config=fl.server.ServerConfig(num_rounds=10), #giá trị num_round
    )
    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5), #giá trị num_round
    )
    #Import svm model
