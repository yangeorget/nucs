import multiprocessing
import random

FIND = 50
MAX_COUNT = 100000

class A:
    def __init__(self):
        self.l = []

class B:
    def __init__(self):
        self.a = A()


def find(process, initial, return_dict, run):
    print(f"{process} find")
    while run.is_set():
        start = initial
        while start <= MAX_COUNT:
            if FIND == start:
                return_dict[process] = f"Found: {process}, start: {initial}"
                run.clear() # Stop running.
                break
            start += random.randrange(0, 10)
            print(f"{process} {start}")


if __name__ == "__main__":
    processes = []
    manager = multiprocessing.Manager()
    return_code = manager.dict()
    run = manager.Event()
    run.set()  # We should keep running.
    for i in range(10):
        process = multiprocessing.Process(
            target=find, args=(f"computer_{i}", i, return_code, run)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    print(return_code.values())