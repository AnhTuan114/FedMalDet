import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True, # prevent refreshing weights when fitting
    )
    model1 = LinearSVC(
        max_iter=1,
        # local epoch
    )

    # Setting initial parameters, akin to model.compile for keras models
    #cập nhật tham số vào mô hìnhq
    utils.set_initial_params(model)
    utils.set_initial_params1(model1)

    # Define Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore #config các tham số cấu honhfmáy chủ yêu cầu. Báo cho client :tham số cần thiết ,thuộc tính vô hướng
            return utils.get_model_parameters(model)#Tham số mô hình cục bộ dưới dạng danh sách nd.arrays


        def get_parameters1(self, config):  # type: ignore #config các tham số cấu honhfmáy chủ yêu cầu. Báo cho client :tham số cần thiết ,thuộc tính vô hướng
            return utils.get_model_parameters1(model1)#Tham số mô hình cục bộ dưới dạng danh sách nd.arrays
        
#config: tham số cấu hình cho phép của máy chủ: truyền giá trị tùy ý từ máy chủ đến máy khách VD: epoch: nút
        def fit(self, parameters, config):  # parameter: tham số mô hình toàn cầu hiện tại, 
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def fit1(self, parameters, config):  # parameter: tham số mô hình toàn cầu hiện tại, 
            utils.set_model_params1(model1, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model1.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters1(model1), len(X_train), {}
        
        def evaluate(self, parameters, config):  # đánh giá mô hình
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
        
        def evaluate1(self, parameters, config):  # đánh giá mô hình
            utils.set_model_params1(model1, parameters)
            loss = log_loss(y_test, model1.predict(X_test))
            accuracy = model1.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client())#khởi động flower NumpyClient/ MnistClient: lớp cơ sở trừu tượng