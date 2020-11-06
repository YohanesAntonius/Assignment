from __future__ import print_function
from ortools.linear_solver import pywraplp
def create_data_model():
    data = {}
    data['constraint_coeffs'] = [
      [0.14, 0.2 , 0.125],
      [240, 360 , 200],
      [8, 10 , 8],
      [4, 4, 4],
      [2, 3, 3],
      [4, 6, 5],
      [0.1, 0.2, 0.1],
      [0.1, 0.1, 0.1],
      [0.3, 0.4, 0.3],
      [1, 1, 1],
      [2, 3, 2],
      [250, 350, 250],
      [400, 600, 400],
      [150, 200, 150],
      [0.5, 0.6, 0.5],
      [0.25, 0.3, 0.25],
      [655003, 855622, 625556],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
  ]
    data['bounds'] = [12, 76000, 600, 500,240, 480,25,25,75,100,200,27200,40000,20000,40,20,65000000,29,24,27]
    data['obj_coeffs'] = [544997, 644378, 574444]
    data['num_vars'] = 3
    data['num_constraints'] = 20
    return data

def main():
  data = create_data_model()
  # Create the mip solver with the CBC backend.
  solver = pywraplp.Solver.CreateSolver('simple_mip_program', 'CBC')
  infinity = solver.infinity()
  x = {}
  for j in range(data['num_vars']):
    x[j] = solver.IntVar(0, infinity, 'x[%i]' % j)
  print('Number of variables =', solver.NumVariables())

  for i in range(data['num_constraints']):
    constraint = solver.RowConstraint(0, data['bounds'][i], '')
    for j in range(data['num_vars']):
      constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
  print('Number of constraints =', solver.NumConstraints())
  # In Python, you can also set the constraints as follows.
  # for i in range(data['num_constraints']):
  #  constraint_expr = \
  # [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
  #  solver.Add(sum(constraint_expr) <= data['bounds'][i])

  objective = solver.Objective()
  for j in range(data['num_vars']):
    objective.SetCoefficient(x[j], data['obj_coeffs'][j])
  objective.SetMaximization()
  # In Python, you can also set the objective as follows.
  # obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
  # solver.Maximize(solver.Sum(obj_expr))

  status = solver.Solve()

  if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in range(data['num_vars']):
      print(x[j].name(), ' = ', x[j].solution_value())
    print()
  else:
    print('The problem does not have an optimal solution.')

if __name__ == '__main__':
  main()
