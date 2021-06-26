import os

import toml

from test_common import make_source_dic
from rcwa.tmm import tmm_

def make_layer_dic(epsilon, mu, thickness):
    return {'epsilon': epsilon, 'mu': mu, 'thickness': thickness}

def test_benchmark():
    '''Test case from Computational Electromagnetics Course Assignment by Raymond Rumpf'''
    try:
        os.remove('output.toml')
    except FileNotFoundError:
        pass
    source_dic = make_source_dic(1, 57, 23, [1, 0], [0, 1])
    superstrate_dic = {'mu': 1.2, 'epsilon': 1.4}
    layer_1_dic = make_layer_dic(2, 1, 0.25)
    layer_2_dic = make_layer_dic(1, 3, 0.5)
    substrate_dic = {'mu': 1.6, 'epsilon': 1.8}

    input_toml = {'layer': [layer_1_dic, layer_2_dic], 'source': source_dic,\
            'superstrate': superstrate_dic, 'substrate': substrate_dic}
    tmm_(input_toml)
    output_toml = toml.load('output.toml')
    assert output_toml['R']['00'] == 0.4403
    assert output_toml['T']['00'] == 0.5597
    assert output_toml['R_T']['00'] == 1
