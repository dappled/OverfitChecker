import itertools
import numpy as np
from collections import Counter
from sklearn import linear_model
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
from concurrent import futures
import functools


def second_step(m, s):
    # dfs = []
    # for g, df in m.groupby(np.arange(len(m)) // (len(m) / s)):
    #     dfs.append(df)
    # return dfs
    return np.split(m, s)


def third_step(s):
    # sets_ids = []
    full_sets = tuple(range(s))
    # full_sets_set = set(full_sets)
    return itertools.combinations(full_sets, int(s / 2))
    # sets_ids.append(item, tuple(full_sets_set ^ set(item))))
    # return sets_ids


def fourth_step_one_time_a_b(sub_dfs, s, separate_set_id):
    # train_set = pd.concat(sub_dfs[idx] for idx in separate_set_id[0])
    # test_set = pd.concat(sub_dfs[idx] for idx in separate_set_id[1])
    train_set = []
    test_set = []
    for i in range(s):
        if i in separate_set_id:
            np.concatenate((train_set, sub_dfs[i]), axis=0)
        else:
            np.concatenate((test_set, sub_dfs[i]), axis=0)
    return train_set, test_set


def fourth_step_one_time_c_d(train_set, test_set, time_len):
    # train_performance = train_set.apply(eval_performance, freq)
    # train_rank = train_performance.rank(method='dense')
    # test_performance = test_set.apply(eval_performance, freq)
    # test_rank = test_performance.rank(method='dense')
    train_performance = np.apply_along_axis(eval_performance, 0, train_set, time_len)
    train_rank = get_rank(train_performance)
    test_performance = np.apply_along_axis(eval_performance, 0, test_set, time_len)
    test_rank = get_rank(test_performance)
    return train_performance, test_performance, train_rank, test_rank


def get_rank(array):
    temp = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    return ranks


def eval_performance(pnl, time_len):
    # freq = float(len(pnl)) / ((pnl.index[-1] - pnl.index[0]).days + 1) * 365.25
    freq2 = np.count_nonzero(pnl) / time_len * 255
    m1 = np.mean(pnl)
    m2 = np.std(pnl)
    # m3 = ss.skew(pnl)
    # m4 = ss.kurtosis(pnl, fisher=False)
    sr = m1 / m2 * freq2 ** .5
    return sr


def fourth_step_one_time_e_f_g(train_rank, test_rank):
    # step e
    best_is_idx = np.argmax(train_rank)
    # step f
    relative_rank_oos = test_rank[best_is_idx] / (train_rank.shape[0] + 1)
    # step g
    logit = np.log(relative_rank_oos / (1 - relative_rank_oos))

    return best_is_idx, logit


def fourth_step_all_together(combination, dfs, freq, s):
    train_set, test_set = fourth_step_one_time_a_b(dfs, s, combination)
    train_performance, test_performance, train_rank, test_rank = fourth_step_one_time_c_d(train_set, test_set, freq)
    best_is_idx, logit = fourth_step_one_time_e_f_g(train_rank, test_rank)
    # best train/test
    best_is_performance = train_performance[best_is_idx]
    best_is_oos_performance = test_performance[best_is_idx]
    return logit, best_is_performance, best_is_oos_performance


def fifth_step(logits):
    c = Counter(logits)
    relative_freq = {cc: c[cc] / len(c) for cc in c}
    return relative_freq


def PBO(relative_freq):
    # pbo is the rate at which optimal is strategies underperform the median of the oos trials
    return sum(v for k, v in relative_freq.items() if k <= 0)


def performance_degradation_prob_of_loss(ax, train_performance, test_performance):
    ax.scatter(train_performance, test_performance, 'r')
    ax.set_xlabel("SR IS")
    ax.set_ylabel("SR OOS")

    # performance degradation
    regr = linear_model.LinearRegression()
    regr.fit(train_performance, test_performance)
    slope = regr.coef_[0]
    intercept = regr.intercept_
    r2 = regr.score()
    ax.plot(train_performance, regr.predict(train_performance))
    ax.text(0.1, 0.9, f"[SR OOS]={intercept}+{slope}*[SR IS]+err | adjR2={r2}")

    # probability of loss
    pol = sum(1 for r in test_performance if r < 0) / len(test_performance)
    ax.text(0.1, 0.1, f"Prob[SR OOS<0]={pol:.2f}", transform=ax.transAxes)
    ax.set_title("OOS Perf. Degradation")


def run_cscv(m, time_len, s):
    if s % 2 != 0:
        raise Exception(f"{s} should be even #")
    if m.shape[0] % s != 0:
        m = m[m.shape[0] % s:]
    dfs = second_step(m, s)
    separate_set_ids = third_step(s)
    logits = []
    best_is_performances = []
    best_is_oos_perfromances = []
    with futures.ProcessPoolExecutor(8) as pool:
        for logit, best_is_performance, best_is_oos_performance in pool.map(functools.partial(fourth_step_all_together, dfs=dfs, time_len=time_len, s=s), separate_set_ids):
            logits.append(logit)
            best_is_performances.append(best_is_performance)
            best_is_oos_perfromances.append(best_is_oos_performance)
    # for combination in separate_set_ids:
    #     train_set, test_set = fourth_step_one_time_a_b(dfs, combination)
    #     train_performance, test_performance, train_rank, test_rank = fourth_step_one_time_c_d(train_set, test_set)
    #     logits.append(fourth_step_one_time_e_f_g(train_rank, test_rank))
    # relative_freq = fifth_step(logits)

    plt.figure()
    ax1 = plt.subplot()

    ax11 = ax1.add_subplot(211)
    performance_degradation_prob_of_loss(ax11, best_is_performances, best_is_oos_perfromances)

    ax12 = ax1.add_subplot(212)
    ax12.hist(logits, weights=np.zeros_like(logits) + 1. / len(logits))
    ax12.set_title("Hist. of Rank Logits")
    ax12.set_xlabel("Logits")
    ax12.set_ylabel("Frequency")

    plt.show()

    # stochastic dominance
    ax2 = plt.subplot()
    train_performance_sorted = np.sort(best_is_performances)
    p = 1. * np.arange(len(train_performance_sorted)) / (len(train_performance_sorted) - 1)
    test_performance_sorted = np.sort(best_is_oos_perfromances)
    ax2.plt(train_performance_sorted, p, label="optimized")
    ax2.plt(test_performance_sorted, p, label="non-optimized")
    ax2.set_xlabel("SR optimized vs. non-optimized")
    ax2.set_ylabel("Frequency")
    ax2.set_title("OOS Cumul. Dist.")

    plt.show()


if __name__ == '__main__':
    pnl_file = r"pnl.csv"
    pnl = pd.DataFrame.from_csv(pnl_file)
    time_len = (pnl.index[-1] - pnl.index[0]).days + 1
    pnl_np = pnl.as_matrix()
    run_cscv(pnl_np, time_len, 16)
    input("Press Enter to end")
