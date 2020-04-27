from policy_gradient import Policy

policy = Policy(6)

import numpy as np


sample_im = np.random.rand(300, 200, 3).astype(np.uint8)

for _ in range(10):
    action = policy.sample_action(sample_im)
    print(action)
