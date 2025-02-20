import numpy as np
import pickle

class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
        self.istrain = False

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')

        data = {
            "weight" : self.weights,
            "bias": self.bias
        }

        with open('And_result.pkl','wb') as f:
            pickle.dump(data,f)

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        with open('And_result.pkl', 'rb') as f:
            data = pickle.load(f)

        self.weights = data['weight']
        self.bias = data['bias']
        
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)
