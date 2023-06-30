import numpy as np
import random
import os as os
import shutil
from tqdm import tqdm

import torch
from itertools import chain
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import ANN
from client import caller, aggregation
from dataprocessing import save_object, read_object, nn_seq_wind

def main():
    # N: No.of clients
    # Cper: Percentage of clients to be chosen for every communication round
    # K: No.of update steps in the clients
    # B: Batch size
    # R: No.of communication rounds
    # input_dim: Dimension of the input
    # lr: learning rate

    N, Cper, K, B, R = 10, 0.5, 10, 50, 10
    input_dim = 28
    lr = 0.08

    clients = []
    for task in range(1, 2):
        for zone in range(1, 11):
            clients.append("Task" + str(task) + "_W_Zone" + str(zone))

    params = {"N": N, "Cper": Cper, "K": K, "B": B, "R": R, "clients": clients, "input_dim": input_dim, "lr": lr}

    # save dictionary of default parameters into a .pkl file
    save_object(params, "defaults.pkl")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn = ANN(input_dim = input_dim, name = "main", B = B, K = K, lr = lr).to(device)

    for k, v in nn.named_parameters():
        nn.control[k] = torch.zeros_like(v.data)
        nn.delta_control[k] = torch.zeros_like(v.data)
        nn.delta_y[k] = torch.zeros_like(v.data)

    save_object(nn, "main.pkl")
    
    if os.path.exists("clients"):
    	shutil.rmtree("clients")
    
    os.makedirs("clients")
   
    for r in range(R):
        print("-----------------------------------Round " + str(r+1) + "-----------------------------------")
        
        S = random.sample(range(0, K), np.max([int(Cper*K), 1]))
    
        for i in S:
            print("-----------------------------------Client " + str(i) + "-----------------------------------")
            caller(i, r)
    
    #final aggregation
    main = read_object("main.pkl")
    main = aggregation(N, main)
    save_object(main, "main.pkl")
        
    #test
    def test(ann):
        ann.eval()
        _, _, Dte = nn_seq_wind(ann.name, ann.B)
        pred = []
        y = []
        for (seq, target) in tqdm(Dte):
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = ann(seq)
                pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
                y.extend(list(chain.from_iterable(target.data.tolist())))

        pred = np.array(pred)
        y = np.array(y)
        print("mae: ", mean_absolute_error(y, pred), "rmse: ", np.sqrt(mean_squared_error(y, pred)))
        
        return mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))
    
    print("\n\n-------------------Testing the final model on all the clients-------------------\n\n")
    
    mae, rmse = [], []
    
    model = read_object("main.pkl")
    model.eval()
    
    c = clients
    for client in c:
        model.name = client
        x, y = test(model)
        mae.append(x)
        rmse.append(y)
    f = open("mae-decentralized.txt", "w")
    for m in mae:
    	f.write(str(m)+"\n")
    f.close()
    
    f = open("rmse-decentralized.txt", "w")
    for r in rmse:
    	f.write(str(r)+"\n")
    f.close()

if __name__ == '__main__':
    main()
