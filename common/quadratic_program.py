import numpy as np
from typing import Dict, List, Tuple, Union, Type
from qpsolvers import solve_qp


class QP:
    """
    Quadratic Program interface using the Python package qpsolvers (solves currently only problems with inequality
    constraints of the form Ax <= b). The state constraints are updated at each time step.
    """

    def __init__(self, num_states: int, a_d: np.ndarray, b_d: np.ndarray, q: np.ndarray, r: np.ndarray,
                 num_steps: int, solver: str):
        """
        Constructor for a QP problem

        :param num_states: number of states
        :param a_d: discretized A-matrix for a single time step
        :param b_d: discretized B-matrix for a single time step
        :param q: state weight matrix
        :param r: input weight matrix
        :param num_steps: number of time steps
        :param: solver: QP solver name
        :param: x_d: desired state
        """
        self._solver = solver
        self._num_steps = num_steps
        self._num_states = num_states
        self._matrices = self._create_qp_matrices(a_d, b_d, q, r)

    @property
    def matrices(self) -> Dict:
        return self._matrices

    def _create_qp_matrices(self, a_d: np.ndarray, b_d: np.ndarray, q: np.ndarray, r: np.ndarray) -> Dict:
        """
        Initialization of matrices for quadratic program except of matrices depending on state variables

        :param a_d: discretized A-matrix
        :param b_d: discretized B-matrix
        :param q: state weight matrix
        :param r: input weight matrix
        :returns matrices for quadratic and matrices for constraints
        """
        a = np.zeros((self._num_states, self._num_states * self._num_steps))
        for i in range(self._num_steps):
            a[:, self._num_states * i:self._num_states * (i + 1)] = np.transpose(np.linalg.matrix_power(a_d, (i + 1)))
        a = a.transpose()

        b = np.zeros((self._num_states * self._num_steps, self._num_steps))
        for i in range(self._num_steps):
            for j in range(i + 1):
                a_n = np.linalg.matrix_power(a_d, ((i + 1) - (j + 1)))
                b[self._num_states * i: self._num_states * (i + 1), j] = np.matmul(a_n, b_d).transpose()

        rr = np.zeros((self._num_steps, self._num_steps))
        for i in range(self._num_steps):
            rr[i, i] = r

        qq = np.zeros((self._num_states * self._num_steps, self._num_states * self._num_steps))
        for i in range(self._num_steps):
            qq[self._num_states * i:self._num_states * (i + 1), self._num_states * i:self._num_states * (i + 1)] = q

        g = np.matmul(qq, b)
        h = np.matmul(b.transpose(), g) + rr

        matrices = {"A": a, "B": b, "G": g, "P": h}
        return matrices

    def create_constraint_matrices(self, min_input: Union[float, int, None], max_input: Union[float, int, None],
                                   constraints: Tuple[Tuple[Type[Union[float, int, None]],
                                                            Type[Union[float, int, None]]], ...] = None):
        """
        Creates constraint matrices of QP problem

        :param min_input: lower bound of input; if no lower bound exist -> None
        :param max_input: upper bound of input; if no upper bound exist -> None
        :param constraints: definition of constraint structure
        """
        if min_input is not None:
            lb = np.ones((self._num_steps, 1)) * min_input
            self.matrices["lb"] = lb
        if max_input is not None:
            ub = np.ones((self._num_steps, 1)) * max_input
            self.matrices["ub"] = ub

        diag_one_matrix_1 = np.zeros((self._num_steps, self._num_steps))
        diag_one_matrix_2 = np.zeros((self._num_steps, self._num_steps))
        np.fill_diagonal(diag_one_matrix_1, -1)
        np.fill_diagonal(diag_one_matrix_2, 1)

        ax = None
        for k in range(len(constraints)):
            constr_state = np.zeros((self._num_steps, self._num_steps * self._num_states))
            selector = k
            for i in range(constr_state.shape[0]):
                constr_state[i, selector] = 1
                selector += self._num_states
            self.matrices["constr_state_" + str(k + 1)] = constr_state
            if ax is None:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    ax = np.concatenate((-np.matmul(constr_state, self.matrices.get("B")),
                                         np.matmul(constr_state, self.matrices.get("B"))), 0)
                elif constraints[k][0] is not None:
                    ax = -np.matmul(constr_state, self.matrices.get("B"))
                elif constraints[k][1] is not None:
                    ax = np.matmul(constr_state, self.matrices.get("B"))
            else:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    ax = np.concatenate((ax, -np.matmul(constr_state, self.matrices.get("B")),
                                         np.matmul(constr_state, self.matrices.get("B"))), 0)
                elif constraints[k][0] is not None:
                    ax = np.concatenate((ax, -np.matmul(constr_state, self.matrices.get("B"))), 0)
                elif constraints[k][1] is not None:
                    ax = np.concatenate((ax, np.matmul(constr_state, self.matrices.get("B"))), 0)

        ax = np.concatenate((ax, diag_one_matrix_1, diag_one_matrix_2), 0)
        self.matrices["Ax"] = ax

    def update_qp_matrices_const(self, x_0: np.ndarray, single_x_d: List[float],
                                 constraints: Tuple[Tuple[Type[Union[float, int, None]],
                                                          Type[Union[float, int, None]]], ...] = None):
        """
        Updates matrices of QP problem. Each time step of a state variable is assigned the same value.

        :param x_0: initial state
        :param single_x_d: desired state for a single time step
        :param constraints: list of tuples containing lower and upper constraints for the state variables
        """
        x_d = np.zeros((self._num_steps * self._num_states, 1))
        x_d[0::3] += single_x_d[0]
        x_d[1::3] += single_x_d[1]

        q = np.matmul((np.matmul(self.matrices.get("A"), x_0) - x_d).transpose(),
                      self.matrices.get("G")).transpose()

        b = None
        for k in range(self._num_states):
            if b is None:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    b = np.concatenate((-constraints[k][0] +
                                        np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                            self.matrices.get("A")), x_0),  # state lower bound
                                        constraints[k][1] - np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state upper bound
                elif constraints[k][0] is not None:
                    b = -constraints[k][0] + np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                                 self.matrices.get("A")), x_0)   # state lower bound
                elif constraints[k][1] is not None:
                    b = constraints[k][1] - np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                                self.matrices.get("A")), x_0)  # state upper bound
            else:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    b = np.concatenate((b,
                                        -constraints[k][0] + np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0),  # state lower bound
                                        constraints[k][1] - np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state upper bound
                elif constraints[k][0] is not None:
                    b = np.concatenate((b,
                                        -constraints[k][0] + np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state lower bound
                elif constraints[k][1] is not None:
                    b = np.concatenate((b, constraints[k][1]
                                        - np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                              self.matrices.get("A")), x_0)), 0)  # state upper bound

        # add constraint for input lower and upper bound
        if self.matrices.get("lb") is not None:
            b = np.concatenate((b, -self.matrices.get("lb")), 0)
        if self.matrices.get("lb") is not None:
            b = np.concatenate((b, self.matrices.get("ub")), 0)
        self.matrices["q"] = q
        self.matrices["b"] = b

    def update_qp_matrices_dyn(self, x_0: np.ndarray, single_x_d: List[float],
                               constraints: Tuple[Tuple[np.array, np.array], ...] = None):
        """
        Updates matrices of QP problem. Each time step of a state variable can be assigned a different value.

        :param x_0: initial state
        :param single_x_d: desired state for a single time step
        :param constraints: list of tuples containing lower and upper constraints for the state variables
        """
        x_d = np.zeros((self._num_steps * self._num_states, 1))
        x_d[0::3] += single_x_d[0]
        x_d[1::3] += single_x_d[1]

        q = np.matmul((np.matmul(self.matrices.get("A"), x_0) - x_d).transpose(),
                      self.matrices.get("G")).transpose()

        b = None
        for k in range(self._num_states):
            if b is None:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    b = np.concatenate((-1 * constraints[k][0] +
                                        np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                            self.matrices.get("A")), x_0),  # state lower bound
                                        constraints[k][1] - np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state upper bound
                elif constraints[k][0] is not None:
                    b = -1 * constraints[k][0] \
                        + np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                              self.matrices.get("A")), x_0)   # state lower bound
                elif constraints[k][1] is not None:
                    b = constraints[k][1] - np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                                self.matrices.get("A")), x_0)  # state upper bound
            else:
                if constraints[k][0] is not None and constraints[k][1] is not None:
                    b = np.concatenate((b,
                                        -1 * constraints[k][0] + np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0),  # state lower bound
                                        constraints[k][1] - np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state upper bound
                elif constraints[k][0] is not None:
                    b = np.concatenate((b,
                                        -1 * constraints[k][0] + np.matmul(
                                            np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                      self.matrices.get("A")), x_0)), 0)  # state lower bound
                elif constraints[k][1] is not None:
                    b = np.concatenate((b, constraints[k][1]
                                        - np.matmul(np.matmul(self.matrices.get("constr_state_" + str(k + 1)),
                                                              self.matrices.get("A")), x_0)), 0)  # state upper bound

        # add constraint for input lower and upper bound
        if self.matrices.get("lb") is not None:
            b = np.concatenate((b, -self.matrices.get("lb")), 0)
        if self.matrices.get("lb") is not None:
            b = np.concatenate((b, self.matrices.get("ub")), 0)
        self.matrices["q"] = q
        self.matrices["b"] = b

    def solve(self):
        """
        Solves quadratic program by calling qpsolvers package
        Quadratic programs of the form:
        min_x 1/2 x^T P x + q^T x
        subject to Gx <= h
        Ax = b
        lb <= x <= ub

        :returns solution for optimization problem
        """
        solution = solve_qp(self._matrices.get("P"), self._matrices.get("q").reshape((-1,)), self._matrices.get("Ax"),
                            self._matrices.get("b").reshape((-1,)), np.zeros(self._matrices.get("Ax").shape),
                            np.zeros((self._matrices.get("b").shape[0],)), solver=self._solver)

        return solution
