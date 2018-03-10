from random import random, choice
import sys
from time import perf_counter
import queue as Queue
import argparse
# import numba
import pickle
from numpy import linspace
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Rollout example')
parser.add_argument('--example', action='store_true',help='run with a given tree')
parser.add_argument('--prob', default=0.5, type=float, help='specify the probability with which a node is inactive')
parser.add_argument('--height', default=6, type=int, help='height of the binary tree')
parser.add_argument('--simulation', action='store_true', help='run the simulation over a number of trials')
parser.add_argument('--prob-steps', default=10, type=int)
parser.add_argument('--height-steps', default=5, type=int)
parser.set_defaults(example=False)
parser.set_defaults(simulation=False)
args = parser.parse_args()


class Node:
    def __init__(self, data, left=None, right=None, parent=None):
        self.data = data
        self.left = left
        self.right = right
        self.parent = parent

    def is_root(self):
        return bool(self.parent is not None)

    def is_leaf(self):
        return bool(self.left is None) and bool(self.right is None)


class BinaryTree:

    def __init__(self, node):
        self.root = node


class SubRes:
    def __init__(self,m,n):
        self.time = np.zeros((m,n))
        self.acc = np.zeros((m,n))


class Result:

    def __init__(self,m,n):
        self.greedy = SubRes(m,n)
        self.dp = SubRes(m,n)
        self.rollout = SubRes(m,n)


def inorder(root):
    stack = Queue.LifoQueue()
    node = root

    done = 0
    while done == 0:

        if node is not None:
            stack.put(node)
            node = node.left
        else:
            if not stack.empty():
                node = stack.get()
                print(node.data)
                node = node.right
            else:
                done = 1


def depth_first(root):
    stack = Queue.LifoQueue()
    node = root
    stack.put(node)
    while not stack.empty():
        node = stack.get()
        print(node.data)
        if node.right is not None:
            stack.put(node.right)
        if node.left is not None:
            stack.put(node.left)


def to_list(root):
    queue = Queue.Queue()
    node = root
    queue.put(node)
    l = []
    while not queue.empty():
        node = queue.get()
        l.append(node.data)
        if node.left is not None:
            queue.put(node.left)
        if node.right is not None:
            queue.put(node.right)
    return l


def breadth_first(root):
    queue = Queue.Queue()
    node = root
    queue.put(node)
    while not queue.empty():
        node = queue.get()
        print(node.data)
        if node.left is not None:
            queue.put(node.left)
        if node.right is not None:
            queue.put(node.right)


def create_binary_tree(height, prob, load_data=True):

    if load_data:
        dd = list(reversed([True, True, True, False, False, True, True]))
        data = dd.pop()
    else:
        data = bool(random() > prob)
    # data = choice(list(range(100)))
    root = Node(data)
    queue = Queue.Queue()
    queue.put(root)
    for i in range(1, height):
        count = 0
        assert(queue.qsize()==2**(i-1))
        while count < 2**(i-1):
            node = queue.get()
            # data = choice(list(range(100)))

            if load_data:
                data = dd.pop()
            else:
                data = bool(random() > prob)

            node.left = Node(data, parent=node)

            # data = choice(list(range(100)))
            if load_data:
                data = dd.pop()
            else:
                data = bool(random() > prob)

            node.right = Node(data, parent=node)

            queue.put(node.left)
            queue.put(node.right)
            count += 1

    return BinaryTree(root)


def greedy_policy(node):

    if node.data is False:
        return False

    while node is not None:
        if not node.is_leaf():
            if node.left.data is True:
                node = node.left

            elif node.right.data is True:
                node = node.right

            else:
                return False
        else:
            if node.data:
                return True
            else:
                return False


def dp_approach(root, n):

    l = to_list(root)
    table = [None] * 2**n
    ind = n
    while ind > 1:
        for i in range(2**(ind-1)-1, 2**ind-1):
            # print(i)
            if 2*i+1 >= 2**n-1:
                l[i] = l[i]
            else:
                # print(i, 2*i+1, 2**n, len(l))
                l[i] = l[i] and (l[2*i+1] or l[2*i+2])

        ind = ind - 1
    res = l[0] and (l[1] or l[2])
    return res


def rollout(tree):

    node = tree.root
    if node.data is False:
        return False

    while node is not None:
        if not node.is_leaf():
            left = node.left
            right = node.right
            if left.data and right.data:
                left_path = greedy_policy(left)
                right_path = greedy_policy(right)
                if left_path or right_path:
                    return True
                else:
                    node = left     # like in greedy policy
            elif left.data:
                node = left
            elif right.data:
                node = right
            else:
                return False
        else:
            return node.data


def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def file_open(filename):
    with open(filename, 'rb') as output:
        ret = pickle.load(output)

    return ret


# @numba.jit(nopython=True, parallel=True)
def run_simulation():
    print(args)
    # result = dict()
    # result['greedy'] = dict()
    # result['dp'] = dict()
    # result['rollout'] = dict()
    #
    # result['greedy']['time'] = []
    # result['greedy']['acc'] = []
    #
    # result['rollout']['time'] = []
    # result['rollout']['acc'] = []
    #
    # result['dp']['time'] = []
    # result['dp']['acc'] = []

    prob_steps = args.prob_steps
    height_steps = args.height_steps
    result = Result(prob_steps, height_steps)
    prob_values = linspace(0.1, 0.9, prob_steps)
    height_values = linspace(6, 8, height_steps).astype(int)
    ref_values = dict()
    ref_values['heights'] = height_values
    ref_values['probs'] = prob_values

    sim_start = perf_counter()

    for prob_ind in range(prob_steps):
        prob = prob_values[prob_ind]

        for height_ind in range(height_steps):
            n = height_values[height_ind]
            n_lim = max(2 ** (n + 1), 1000)

            for _ in tqdm.tqdm(range(n_lim)):

                tree = create_binary_tree(n, prob, load_data=False)

                greedy_start = perf_counter()
                greedy_ans = greedy_policy(tree.root)
                greedy_stop = perf_counter()

                dp_start = perf_counter()
                dp_ans = dp_approach(tree.root, n)
                dp_stop = perf_counter()

                rollout_start = perf_counter()
                rollout_ans = rollout(tree)
                rollout_stop = perf_counter()

                result.greedy.time[prob_ind][height_ind] += greedy_stop-greedy_start
                result.dp.time[prob_ind][height_ind] += dp_stop-dp_start
                result.rollout.time[prob_ind][height_ind] += rollout_stop-rollout_start

                result.greedy.acc[prob_ind][height_ind] += 1 if greedy_ans == dp_ans else 0
                result.dp.acc[prob_ind][height_ind] = 1
                result.rollout.acc[prob_ind][height_ind] += 1 if rollout_ans == dp_ans else 0

            result.greedy.acc[prob_ind][height_ind]/=n_lim
            result.rollout.acc[prob_ind][height_ind]/=n_lim

            result.greedy.time[prob_ind][height_ind] /= n_lim
            result.rollout.time[prob_ind][height_ind] /= n_lim
            result.dp.time[prob_ind][height_ind] /= n_lim

            sim_stop = perf_counter()

            print(sim_stop - sim_start)

    save(result, 'result.pkl')
    save(ref_values, 'reference.pkl')


def main():
    if not args.simulation:
        print(args)
        if args.example:
            n = 3
        else:
            n = args.height
        prob = args.prob
        tree = create_binary_tree(n, prob, load_data=args.example)
        if args.example:
            print("inorder traversal")
            inorder(tree.root)
            print("breadth first traversal")
            breadth_first(tree.root)

        greedy_start = perf_counter()
        print("greedy policy:", greedy_policy(tree.root))
        greedy_stop = perf_counter()
        print("time taken: {}".format(greedy_stop-greedy_start))

        dp_start = perf_counter()
        print("dp exact:", dp_approach(tree.root, n))
        dp_end = perf_counter()
        print("time taken: {}".format(dp_end-dp_start))

        rollout_start = perf_counter()
        print("rollout:", rollout(tree))
        rollout_end = perf_counter()
        print("time taken: {}".format(rollout_end-rollout_start))

    else:
        run_simulation()


if __name__=="__main__":
    main()

