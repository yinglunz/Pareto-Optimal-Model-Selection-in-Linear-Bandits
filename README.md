# Pareto Optimal Model Selection in Linear Bandits

This is the python code for our AISTATS 2022 paper **Pareto Optimal Model Selection in Linear Bandits**. Packages used include: numpy, enum, math, copy, multiprocessing, pickle, time and matplotlib. 

We only include our implementations of `LinUCB`, `LinUCB Oracle`, `Dynamic Balancing`, (part of) `LinUCB++ with Carrol`, and `LinUCB++`. Please contact authors of `Smooth Corral` regarding the implement of `Smooth Corral`.

Use the following commands to reproduce experiments in Figure 1.

```
python3 regret_curve.py
python3 regret_wrt_alpha.py
python3 plot_curve.py
python3 plot_wrt_alpha.py
```

Other experiment results can be reproduced in a similar way, with appropriate changes of parameters `expressiveness, d, K, theta_star_oracle` in `regret_curve.py`.

On a cluster consists of two Intel® Xeon® Gold 6254 Processors, the runtime for `pyhton3 regret_curve.py` is around 1 hour and the runtime for `pyhton3 regret_wrt_alpha.py` is around 5.5 hours.
