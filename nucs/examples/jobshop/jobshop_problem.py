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
from typing import List, Optional

from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_CRITICAL_RESOURCE
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ADD_C_EQ, ALG_DISJUNCTIVE, ALG_LINEAR_LEQ_C, ALG_MAX_EQ
from nucs.solvers.backtrack_solver import Search


class JobShopProblem(Problem):
    """
    The Job-Shop Scheduling Problem.

    There are n jobs and m machines. Each job is a chain of m operations that must run in order; operation k
    of a job runs on a given machine for a given duration. Each machine processes one operation at a time.
    The objective is to minimize the makespan (the completion time of the last operation).

    The model has one start-time variable per operation, one completion-time variable per job and a makespan
    variable. Operations of the same job are chained by precedence constraints, the operations of each machine
    are constrained by a global disjunctive (no-overlap) constraint, and the makespan is the maximum of the
    job completion times (so it is fully determined once the start times are bound).

    Instances follow the OR-Library format: ``jobs[j]`` is the list of ``[machine, duration]`` operations of
    job j, in processing order.

    :meth:`recommended_searches` returns the search to drive a :class:`BacktrackSolver` with.
    """

    def __init__(self, jobs: List[List[List[int]]], horizon: Optional[int] = None) -> None:
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
        if horizon is None:
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
        # one completion-time variable per job (the completion of its last operation) and the makespan
        completion_domains = [(sum(durations[j]), horizon) for j in range(self.job_nb)]
        super().__init__(start_domains + completion_domains + [(makespan_lb, horizon)])
        self.completion_start = self.job_nb * self.machine_nb
        self.makespan = self.completion_start + self.job_nb
        # precedence: operation k must finish before operation k+1 of the same job starts
        for j in range(self.job_nb):
            for k in range(self.machine_nb - 1):
                self.add_propagator(
                    ALG_LINEAR_LEQ_C, [self.start(j, k), self.start(j, k + 1)], [1, -1, -durations[j][k]]
                )
            # completion[j] = start of the last operation + its duration
            last = self.machine_nb - 1
            self.add_propagator(ALG_ADD_C_EQ, [self.start(j, last), self.completion_start + j], [durations[j][last]])
        # makespan = max of the job completion times (determined once the starts are bound)
        self.add_propagator(
            ALG_MAX_EQ, list(range(self.completion_start, self.completion_start + self.job_nb)) + [self.makespan]
        )
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

    def resource_params(self) -> List[List[int]]:
        """
        Returns the ``(machine, duration)`` of each start-time variable, in variable-index order.

        This is the parameter array consumed by the critical-resource variable heuristic: it tells the
        heuristic which machine every task belongs to and how long it runs, so it can focus branching on one
        critical machine at a time.

        :return: one ``[machine, duration]`` row per start-time variable
        :rtype: List[List[int]]
        """
        return [[op[0], op[1]] for job in self.jobs for op in job]

    def jobshop_searches(self) -> List[Search]:
        """
        Returns the recommended search for this problem.

        It branches only on the start times, sequencing one critical machine at a time (the critical-resource
        variable heuristic) and committing each selected task to its earliest start (the min_value domain
        heuristic) -- a cheap, backtrack-light schedule-earliest strategy and, of NuCS's job-shop searches, the
        strongest at proving optimality. Pass it to a solver with ``BacktrackSolver(problem,
        searches=problem.recommended_searches())``.

        :return: a single-search sequential search
        :rtype: List[Search]
        """
        return [
            Search(
                decision_variables=range(self.completion_start),
                var_heuristic=VAR_HEURISTIC_CRITICAL_RESOURCE,
                var_heuristic_params=self.resource_params(),
                dom_heuristic=DOM_HEURISTIC_MIN_VALUE,
            )
        ]

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
