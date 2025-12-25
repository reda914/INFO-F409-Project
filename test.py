import numpy as np
import random


def get_action_rules(action_rule):
    # Convert rule_id to 4-bit binary
    bits = [(action_rule >> i) & 1 for i in range(4)]  # bits[0]=LSB, bits[3]=MSB
    return bits

print(get_action_rules(5))