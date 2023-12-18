from part_Alex import *
from sys import argv

L, m_warmup, m_sweep_list = int(argv[1]), int(argv[2]), list(map(int, argv[3:]))
finite_system_algorithm(L, m_warmup, m_sweep_list)