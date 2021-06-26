from rcwa.common import get_input
from rcwa.rcwa import rcwa_
from rcwa.tmm import tmm_

def rcwa():
    input_toml = get_input()
    rcwa_(input_toml)

def tmm():
    input_toml = get_input()
    tmm_(input_toml)
