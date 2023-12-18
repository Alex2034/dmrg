import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh 

from part_Sasha import *

model_d = 2  # Single-site basis size

Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # Single-site S^z
Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # Single-site S^+
H1 = np.array([[0, 0], [0, 0]], dtype='d')  # Single-site portion of H is zero

def single_dmrg_step(sys, env, m):
    assert is_valid_block(sys) and is_valid_block(env)

    sys_enl, env_enl = enlarge_blocks(sys, env)
    assert is_valid_enlarged_block(sys_enl) and is_valid_enlarged_block(env_enl)

    superblock_hamiltonian = construct_superblock_hamiltonian(sys_enl, env_enl)
    energy, psi0 = calculate_ground_state(superblock_hamiltonian)
    rho = reduced_density_matrix(sys_enl, psi0)

    evals, evecs = np.linalg.eigh(rho)
    eigenstates = sort_eigenstates(evals, evecs)
    transformation_matrix, my_m = build_transformation_matrix(eigenstates, m, sys_enl)

    new_operator_dict = rotate_truncate_operators(transformation_matrix, sys_enl)
    newblock = Block(length=sys_enl.length, basis_size=my_m, operator_dict=new_operator_dict)

    return newblock, energy

def graphic_representation(sys_block, env_block, system_on_left=True):
    """Returns a graphical representation of the DMRG step."""
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    return graphic if system_on_left else graphic[::-1]

def initialize_blocks(L, m_warmup, initial_block):
    """Initialize blocks using the infinite system algorithm."""
    block_disk = {}
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block

    while 2 * block.length < L:
        print(graphic_representation(block, block))
        block, energy = single_dmrg_step(block, block, m=m_warmup)
        print("E/L =", energy / (block.length * 2))
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    return block_disk, block

def perform_sweeps(L, m_sweep_list, block_disk, initial_sys_block):
    """Perform sweeps using the finite system algorithm."""
    sys_label, env_label = "l", "r"
    sys_block = initial_sys_block

    for m in m_sweep_list:
        while True:
            env_block = block_disk[env_label, L - sys_block.length - 2]
            if env_block.length == 1:
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            print(graphic_representation(sys_block, env_block, sys_label == "l"))
            sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)
            print("E/L =", energy / L)
            block_disk[sys_label, sys_block.length] = sys_block

            if sys_label == "l" and 2 * sys_block.length == L:
                break

def finite_system_algorithm(L, m_warmup, m_sweep_list):
    assert L % 2 == 0  # Require that L is an even number

    block_disk, last_block = initialize_blocks(L, m_warmup, initial_block)
    perform_sweeps(L, m_sweep_list, block_disk, last_block)
    
    
