import pandas_datareader as pdr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import argparse

import gurobipy as gp
from gurobipy import GRB

start = '2018-01-01'
#start = '2019-12-01'
end = '2021-01-01'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Modern Portfolio Theory')
    parser.add_argument('--no_short', action='store_true',
                        help='limit short')
    parser.add_argument('--no_gurobi', action='store_true',
                        help='do not use gurobi')
    parser.add_argument('-n', type=int, default=30,
                        help='Number of companies (default: 30)')
    return parser.parse_args()

def minimize_var(v, mu, cov, shortExists=True):
    #print(v)
    #create a model
    m = gp.Model("quadratic")
    #create variables
    if shortExists:
        theta = m.addMVar(mu.shape[0], lb=float('-inf'), ub=float('inf'), name='theta')
    else:
        #theta = m.addMVar(mu.shape[0], lb=0.0, ub=1.0, name='theta')
        theta = m.addMVar(mu.shape[0], lb=0.0, ub=float('inf'), name='theta')

    #set objective
    obj = theta @ cov @ theta
    m.setObjective(obj)

    #add Constraint
    m.addConstr((theta @ mu)-v==0, 'c0')
    m.addConstr(theta.sum()-1==0, 'c1')

    m.optimize()
    return m

def minimize_var_with_rf(v, rf_rate, mu, cov, shortExists=True):
    #create a model
    m = gp.Model("quadratic")
    #create variables
    if shortExists:
        theta = m.addMVar(mu.shape[0], lb=float('-inf'), ub=float('inf'), name='theta')
    else:
        #theta = m.addMVar(mu.shape[0], lb=0.0, ub=1.0, name='theta')
        theta = m.addMVar(mu.shape[0], lb=0.0, ub=float('inf'), name='theta')

    #set objective
    obj = theta @ cov @ theta
    m.setObjective(obj)

    #add Constraint
    m.addConstr( ((mu-rf_rate) @ theta) == v-rf_rate, 'c0' ) 

    m.optimize()
    return m

def load_dataframes(args):
    df = pd.read_csv('./data/sp500_20000101.csv', header=[0, 1])
    df.drop([0], axis=0, inplace=True)  # drop this row because it only has one column with Date in it
    df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')], format='%Y-%m-%d')  # convert the first column to a datetime
    df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)  # set the first column as the index
    df.index.name = None  # rename the index

    mc_list = [] #ascending order
    with open('./data/mc_list_2021_04_26.csv', 'r') as f:
        mc_reader = csv.reader(f)
        for r in mc_reader:
            mc_list.append(r[1])
    mc_list.reverse()

    df = df.Close
    df = df.resample('M').last()
    df = df[start: end]
    log_df = np.log(df)- np.log(df.shift())
    log_df = log_df[log_df.columns[~log_df.loc['2020-01-01':].isnull().any()]]
    #log_df = log_df[log_df.columns[:50]]
    stocks = mc_list[:args.n]
    for n in ['LUMN', 'CARR', 'OTIS']:
        if n in stocks:
            stocks.remove(n)
    log_df = log_df[stocks]

    vix_df = pd.read_csv("./data/VIX_20170101.csv",index_col='DATE')
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_df = vix_df.resample('M').last()[start:end]
    VIX = vix_df.mean().values[0]

    rf_df = pd.read_csv('./data/TBILL.csv', index_col='DATE')
    rf_df = rf_df[start:end]
    rf_rate = np.log(rf_df*0.01+1).mean().values[0]

    return log_df, VIX, rf_rate

def utility(sig_p, UTIL, A):
    return A*sig_p*sig_p/2 + UTIL

def no_short(log_df, A, rf_rate, mu, cov):
    max_y = np.max(mu)

    y_eff = np.linspace(0, max_y, 100).tolist()
    x_eff = [math.sqrt(minimize_var(y, mu, cov, shortExists=False).ObjVal) for y in y_eff]

    y_CML = np.linspace(0, max_y, 100).tolist()
    x_CML = [math.sqrt(minimize_var_with_rf(y, rf_rate, mu, cov, shortExists=False).ObjVal) for y in y_CML]

    slope = (y_CML[-1]-rf_rate)/x_CML[-1]

    sig_P = slope / A
    mu_P = slope*sig_P + rf_rate
    UTIL = mu_P - A *sig_P*sig_P /2

    x_util = np.linspace(0, x_eff[-1], 100).tolist()
    y_util = [utility(x, UTIL, A) for x in x_util]

    m_P = minimize_var_with_rf(mu_P, rf_rate, mu, cov, shortExists=False)
    theta_P = np.array([v.x for v in m_P.getVars()])
    rf_ratio = 1- theta_P.sum()

    #top 5
    k = 5
    idx = np.argpartition(theta_P, -k)
    top_k_names = log_df.columns[idx[-k:]].tolist()
    top_y = mu[idx[-k:]].tolist()
    top_x = [math.sqrt(cov[x][x]) for x in idx[-k:]]

    title = 'no_short'
    plt.plot(x_eff, y_eff, label='Efficient Frontier')
    plt.plot(x_CML, y_CML, label='CML')
    plt.plot(x_util, y_util, label='Utility')
    plt.scatter(sig_P, mu_P)
    plt.scatter(top_x, top_y, label='top_5')
    plt.title(title)
    plt.xlabel('sigma')
    plt.ylabel('return')
    plt.legend()
    for i, name in enumerate(top_k_names):
        plt.annotate(name, (top_x[i], top_y[i]))
    plt.tight_layout
    plt.savefig(title+'.png')

    return rf_ratio, theta_P, mu_P, sig_P
    
def with_short_with_gurobi(log_df, A, rf_rate, mu, cov):
    max_x = np.max(mu)*2
    y_eff = np.linspace(0, max_x, 100).tolist()
    x_eff = [math.sqrt(minimize_var(y, mu, cov, shortExists=True).ObjVal) for y in y_eff]

    y_CML = np.linspace(0, max_x, 100).tolist()
    x_CML = [math.sqrt(minimize_var_with_rf(y, rf_rate, mu, cov, shortExists=True).ObjVal) for y in y_CML]

    slope = (y_CML[-1]-rf_rate)/x_CML[-1]

    sig_P = slope / A
    mu_P = slope*sig_P + rf_rate
    UTIL = mu_P - A *sig_P*sig_P /2

    x_util = np.linspace(0, x_eff[-1], 100).tolist()
    y_util = [utility(x, UTIL, A) for x in x_util]

    m_P = minimize_var_with_rf(mu_P, rf_rate, mu, cov, shortExists=True)
    theta_P = np.array([v.x for v in m_P.getVars()])
    rf_ratio = 1- theta_P.sum()

    #top 5
    k = 5
    idx = np.argpartition(theta_P, -k)
    top_k_names = log_df.columns[idx[-k:]].tolist()
    top_y = mu[idx[-k:]].tolist()
    top_x = [math.sqrt(cov[x][x]) for x in idx[-k:]]

    title = 'with_short_with_gurobi'
    plt.plot(x_eff, y_eff, label='Efficient Frontier')
    plt.plot(x_CML, y_CML, label='CML')
    plt.plot(x_util, y_util, label='Utility')
    plt.scatter(sig_P, mu_P)
    plt.scatter(top_x, top_y, label='top_5')
    plt.title(title)
    plt.xlabel('sigma')
    plt.ylabel('return')
    plt.legend()
    for i, name in enumerate(top_k_names):
        plt.annotate(name, (top_x[i], top_y[i]))
    plt.tight_layout
    plt.savefig(title+'.png')

    return rf_ratio, theta_P, mu_P, sig_P

def with_short_no_gurobi(log_df, A, rf_rate, mu, cov):
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)

    alpha = mu.dot(cov_inv).dot(mu)
    beta = np.sum(cov_inv.dot(mu))
    gamma = np.sum(cov_inv)
    D = alpha*gamma - beta**2
    E = alpha - 2*beta *rf_rate + gamma*rf_rate**2

    def std(v):
        return math.sqrt((gamma*v**2 - 2*beta*v + alpha) / D)

    def CML(risk):
        return rf_rate + math.sqrt(E)*risk

    sig_M = math.sqrt(E)/ (beta - rf_rate*gamma)
    mu_M = rf_rate + E / (beta - rf_rate*gamma)

    theta_M =  1/(beta - gamma*rf_rate)*cov_inv.dot(mu - rf_rate)

    sig_P = math.sqrt(E)/A
    mu_P = rf_rate + math.sqrt(E)*sig_P
    UTIL = mu_P - A*sig_P*sig_P/2

    max_x = max(mu_M, mu_P)

    x_CML = np.linspace(0, max_x, 100).tolist()
    y_CML = [CML(x) for x in x_CML]

    y_eff = np.linspace(0, y_CML[-1], 100).tolist()
    x_eff = [std(y) for y in y_eff]

    x_util = np.linspace(0, max_x, 100).tolist()
    y_util = [utility(x, UTIL, A) for x in x_util]

    rf_ratio = (mu_M - mu_P)/(mu_M - rf_rate)
    stock_ratio = 1- rf_ratio
    theta_P = stock_ratio * theta_M

    #top 5
    k = 5
    idx = np.argpartition(theta_P, -k)
    top_k_names = log_df.columns[idx[-k:]].tolist()
    top_y = mu[idx[-k:]].tolist()
    top_x = [math.sqrt(cov[x][x]) for x in idx[-k:]]

    title = 'with_short_no_gurobi'
    plt.plot(x_eff, y_eff, label='Efficient Frontier')
    plt.plot(x_CML, y_CML, label='CML')
    plt.plot(x_util, y_util, label='Utility')
    plt.scatter(sig_M, mu_M)
    plt.scatter(sig_P, mu_P)
    plt.scatter(top_x, top_y, label='top_5')
    plt.title(title)
    plt.xlabel('sigma')
    plt.ylabel('return')
    plt.legend()
    for i, name in enumerate(top_k_names):
        plt.annotate(name, (top_x[i], top_y[i]))
    plt.tight_layout
    plt.savefig(title+'.png')

    return rf_ratio, theta_P, mu_P, sig_P
    
if __name__=='__main__':
    args = parse_arguments()
    
    log_df, VIX, rf_rate = load_dataframes(args)
    cov = log_df.cov().to_numpy()
    mu = log_df.mean().to_numpy()
    #A = VIX*20

    if args.no_short: #no short with gurobi
        A = VIX*0.1
        rf_ratio, theta_P, mu_P, sig_P = no_short(log_df, A, rf_rate, mu, cov)
    elif args.no_gurobi: #short exists and no gurobi
        A = VIX*15
        rf_ratio, theta_P, mu_P, sig_P = with_short_no_gurobi(log_df, A, rf_rate, mu, cov)
    else: #short exists and with gurobi 
        A = VIX*15
        rf_ratio, theta_P, mu_P, sig_P = with_short_with_gurobi(log_df, A, rf_rate, mu, cov)
    
    print("Number of top market cap companies:", args.n)
    print("Data start:", start)
    print("Data end(excluding):", end)
    print("VIX:", VIX)
    print("Risk free rate:", rf_rate)
    print("Expected portfolio return:", mu_P)
    print("Expected portfolio standard deviation:", sig_P)
    print("Percentage of portfolio to invest in risk free asset:", rf_ratio)
    print("Company list:")
    print(log_df.columns.to_numpy())
    print("Percentage of rest of the portfolio to invest:")
    print(theta_P)




