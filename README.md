# Modeling Other Players with Bayesian Beliefs for Games with Incomplete Information
<div align="center">

[[arXiv]](https://arxiv.org/abs/2405.14122)
[[PDF]](https://arxiv.org/pdf/2405.14122.pdf)

</div>

Bayesian games model interactive decision-making where players have incomplete information – e.g., regarding payoffs and private data on players’ strategies and preferences – and must actively reason and update their belief models (with regard to such information) using observation and interaction history. Existing work on counterfactual regret minimization have shown great success for games with complete or imperfect information, but not for Bayesian games. To this end, we introduced a new CFR algorithm: Bayesian-CFR and analyze its regret bound with respect to Bayesian Nash Equilibria in Bayesian games. First, we present a method for updating the posterior distribution of beliefs about the game and about other players’ types. The method uses a kernel-density estimate and is shown to converge to the true distribution. Second, we define Bayesian regret and present a Bayesian-CFR minimization algorithm for computing the Bayesian Nash equilibrium. Finally, we extend this new approach to other existing algorithms, such as Bayesian-CFR+ and Deep Bayesian CFR. Experimental results show that our proposed solutions significantly outperform existing methods in classical Texas Hold’em games.

# Installation
```
none
```

# Training
To start training, run the following commmands
```bash
python main.py --algo bcfr
```

# Reference

```bibtex
@article{zhang2024modeling,
  title={Modeling other players with bayesian beliefs for games with incomplete information},
  author={Zhang, Zuyuan and Imani, Mahdi and Lan, Tian},
  journal={arXiv preprint arXiv:2405.14122},
  year={2024}
}
```

# Acknowledgements

The codebase is derived from [rlcard](https://github.com/mjiang9/_rlcard).