import pickle
import codecs
import copy
import torch

def weights_average_function(weights):
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg


def obj_to_pickle_string(x, file_path=None, save=False):
    if save and file_path is not None:
        with open(file_path, 'wb') as output:
            pickle.dump(x, output)
        return file_path
    else:
        encoded = codecs.encode(pickle.dumps(x), "base64").decode()
        return encoded

def pickle_file_to_obj(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        # print("load model from file")
        return pickle.load(df)
    else:
        # print("load model from byte")
        return pickle.loads(codecs.decode(s.encode(), "base64"))