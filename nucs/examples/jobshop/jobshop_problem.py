###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_DISJUNCTIVE, ALG_LINEAR_LEQ_C


class JobShopProblem(Problem):
    """
    The Job-Shop Scheduling Problem.

    There are n jobs and m machines. Each job is a chain of m operations that must run in order; operation k
    of a job runs on a given machine for a given duration. Each machine processes one operation at a time.
    The objective is to minimize the makespan (the completion time of the last operation).

    The model has one start-time variable per operation plus a makespan variable. Operations of the same job
    are chained by precedence constraints, the operations of each machine are constrained by a global
    disjunctive (no-overlap) constraint, and the makespan bounds the completion of every job.

    Instances follow the OR-Library format: ``jobs[j]`` is the list of ``[machine, duration]`` operations of
    job j, in processing order.
    """

    def __init__(self, jobs: List[List[List[int]]]) -> None:
        """
        Inits the problem.

        :param jobs: for each job, the list of [machine, duration] operations in processing order
        :type jobs: List[List[List[int]]]
        """
        self.jobs = jobs
        self.job_nb = len(jobs)
        self.machine_nb = len(jobs[0])
        machines = [[op[0] for op in job] for job in jobs]
        durations = [[op[1] for op in job] for job in jobs]
        horizon = sum(sum(job_durations) for job_durations in durations)
        # earliest start = sum of preceding durations in the job (head); latest start = horizon minus the
        # remaining durations from that operation on (tail).
        start_domains = []
        for j in range(self.job_nb):
            head = 0
            tail = sum(durations[j])
            for k in range(self.machine_nb):
                start_domains.append((head, horizon - tail))
                head += durations[j][k]
                tail -= durations[j][k]
        # the makespan cannot be smaller than the longest job nor than the busiest machine
        machine_load = [0] * self.machine_nb
        for j in range(self.job_nb):
            for k in range(self.machine_nb):
                machine_load[machines[j][k]] += durations[j][k]
        makespan_lb = max(max(sum(d) for d in durations), max(machine_load))
        super().__init__(start_domains + [(makespan_lb, horizon)])
        self.makespan = self.job_nb * self.machine_nb
        # precedence: operation k must finish before operation k+1 of the same job starts
        for j in range(self.job_nb):
            for k in range(self.machine_nb - 1):
                self.add_propagator(
                    ALG_LINEAR_LEQ_C, [self.start(j, k), self.start(j, k + 1)], [1, -1, -durations[j][k]]
                )
            # makespan >= completion time of the last operation of the job
            last = self.machine_nb - 1
            self.add_propagator(ALG_LINEAR_LEQ_C, [self.start(j, last), self.makespan], [1, -1, -durations[j][last]])
        # disjunctive: the operations assigned to a machine must not overlap
        for machine in range(self.machine_nb):
            machine_starts = []
            machine_durations = []
            for j in range(self.job_nb):
                for k in range(self.machine_nb):
                    if machines[j][k] == machine:
                        machine_starts.append(self.start(j, k))
                        machine_durations.append(durations[j][k])
            self.add_propagator(ALG_DISJUNCTIVE, machine_starts, machine_durations)

    def start(self, job: int, operation: int) -> int:
        """
        Returns the variable index of the start time of an operation.

        :param job: the job index
        :type job: int
        :param operation: the operation index within the job
        :type operation: int

        :return: the start-time variable index
        :rtype: int
        """
        return job * self.machine_nb + operation
