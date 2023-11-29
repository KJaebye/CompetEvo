import numpy as np
import multiprocessing

num_cpus = multiprocessing.cpu_count()
print("I: the number of cpu cores: {}".format(num_cpus))

def worker(num, randomstate):
    """子进程执行的函数"""
    print('Worker %d started.' % num)
    s = randomstate.randint(0, 10)
    print(s)

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, np.random.RandomState()))
        jobs.append(p)
        p.start()