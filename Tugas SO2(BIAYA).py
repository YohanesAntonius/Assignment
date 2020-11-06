from ortools.sat.python import cp_model


def main():
    # Data
    costs = [
        [340, 137, 770, 640, 210, 882],
        [895, 714, 889, 572, 570, 367],
        [725, 394, 200, 741, 572, 333],
        [746, 274, 567, 227, 746, 248],
        [837, 777, 471, 125, 357, 806],
        [651, 336, 385, 432, 729, 665],
        [588, 222, 430, 181, 120, 160],
        [574, 718, 168, 773, 562, 365],
        [433, 846, 791, 350, 204, 236],
        [526, 731, 823, 831, 176, 512],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])

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
            objective_terms.append(costs[i][j] * x[i][j])
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Total cost = %i' % solver.ObjectiveValue())
        print()
        for i in range(num_workers):
            for j in range(num_tasks):
                if solver.BooleanValue(x[i][j]):
                    print('Worker ', i, ' assigned to task ', j, '  Cost = ',
                          costs[i][j])
    else:
        print('No solution found.')


if __name__ == '__main__':
    main()
