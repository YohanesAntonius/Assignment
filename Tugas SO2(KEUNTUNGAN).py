from ortools.sat.python import cp_model


def main():
    # Data
    profit = [
        [1607, 1627, 1666, 1182, 1298, 1276],
        [1334, 1284, 1208, 1629, 1133, 1137],
        [957,  1302,  986, 1568, 1239, 1623],
        [1825, 1827,  787, 1463,  959, 1133],
        [777,  1809, 1843,  711,  983, 1764],
        [903,  1366,  996, 1640, 1553, 1994],
        [1851, 1313, 1053, 1614, 1968, 1998],
        [1530,  853,  887, 1819, 1534,  886],
        [925,  1312,  798, 1599,  867, 1361],
        [1740, 1644, 1646,  952, 1160,  876],
    ]
    num_workers = len(profit)
    num_tasks = len(profit[0])

    # Model
    model = cp_model.CpModel()

    # Variables
    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewBoolVar('x[%i,%i]' % (i, j)))
        x.append(t)

    # Constraints
    # Each worker is assigned to at most one task.
    for i in range(num_workers):
        model.Add(sum(x[i][j] for j in range(num_tasks)) == 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        model.Add(sum(x[i][j] for i in range(num_workers)) >= 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(profit[i][j] * x[i][j])
    model.Maximize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Total Profit = %i' % solver.ObjectiveValue())
        print()
        for i in range(num_workers):
            for j in range(num_tasks):
                if solver.BooleanValue(x[i][j]):
                    print('Worker ', i, ' assigned to task ', j, '  Profit = ',
                          profit[i][j])
    else:
        print('No solution found.')


if __name__ == '__main__':
    main()
