import pickle

a = [1,2,3]

with open('test.pkl', 'wb') as f:
    pickle.dump(a, f)

with open('test.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print(loaded_data)