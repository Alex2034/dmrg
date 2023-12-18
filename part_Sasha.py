'''
Imports
'''
import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh 


'''
Class of blocks for DMRG
'''
class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

class EnlargedBlock:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

def is_valid_block(block):
    return all(op.shape == (block.basis_size, block.basis_size) for op in block.operator_dict.values())


'''
Global variables
'''
is_valid_enlarged_block = is_valid_block

model_d = 2  # Single-site basis size

Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # Single-site S^z
Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # Single-site S^+
H1 = np.array([[0, 0], [0, 0]], dtype='d')  # Single-site portion of H is zero

initial_block = Block(1, model_d, {"H": H1, "conn_Sz": Sz1, "conn_Sp": Sp1})


'''
Functions
'''
def H2(Sz1, Sp1, Sz2, Sp2):
    J, Jz = 1., 1.
    return (J / 2) * (kron(Sp1, Sp2.T.conj()) + kron(Sp1.T.conj(), Sp2)) + Jz * kron(Sz1, Sz2)

def enlarge_block(block):
    mblock = block.basis_size
    o = block.operator_dict

    enlarged_operator_dict = {
        "H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2(o["conn_Sz"], o["conn_Sp"], Sz1, Sp1),
        "conn_Sz": kron(identity(mblock), Sz1),
        "conn_Sp": kron(identity(mblock), Sp1),
    }

    return EnlargedBlock(block.length + 1, block.basis_size * model_d, enlarged_operator_dict)

def rotate_and_truncate(operator, transformation_matrix):
    T_conj_transpose = transformation_matrix.T.conj()
    return T_conj_transpose @ operator @ transformation_matrix

def enlarge_blocks(sys, env):
    """Enlarge system and environment blocks."""
    sys_enl = enlarge_block(sys)
    env_enl = enlarge_block(env) if sys is not env else sys_enl
    return sys_enl, env_enl

def construct_superblock_hamiltonian(sys_enl, env_enl):
    """Construct the full superblock Hamiltonian."""
    return kron(sys_enl.operator_dict["H"], identity(env_enl.basis_size)) + \
           kron(identity(sys_enl.basis_size), env_enl.operator_dict["H"]) + \
           H2(sys_enl.operator_dict["conn_Sz"], sys_enl.operator_dict["conn_Sp"],
              env_enl.operator_dict["conn_Sz"], env_enl.operator_dict["conn_Sp"])

def calculate_ground_state(superblock_hamiltonian):
    """Calculate the ground state using ARPACK."""
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")
    return energy, psi0

def reduced_density_matrix(sys_enl, psi0):
    """Construct the reduced density matrix of the system."""
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    return np.dot(psi0, psi0.T.conj())

def sort_eigenstates(evals, evecs):
    """Sort eigenvectors by eigenvalue."""
    return sorted(zip(evals, evecs.T), reverse=True, key=lambda x: x[0])

def build_transformation_matrix(eigenstates, m, sys_enl):
    """Build the transformation matrix from the most significant eigenvectors."""
    my_m = min(len(eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (_, evec) in enumerate(eigenstates[:my_m]):
        transformation_matrix[:, i] = evec
    truncation_error = 1 - sum(eigenvalue for eigenvalue, _ in eigenstates[:my_m])
    print("truncation error:", truncation_error)
    return transformation_matrix, my_m

def rotate_truncate_operators(transformation_matrix, sys_enl):
    """Rotate and truncate each operator."""
    new_operator_dict = {name: rotate_and_truncate(op, transformation_matrix)
                         for name, op in sys_enl.operator_dict.items()}
    return new_operator_dict