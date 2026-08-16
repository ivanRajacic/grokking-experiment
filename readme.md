# Reproducing Grokking from Scratch in PyTorch

## 1. Introduction

A model trained on a small algorithmic task first memorizes its training data and then, thousands of epochs after train loss hits zero, abruptly generalizes. It's exactly the delay between memorization and generalization that is the phenomenon - **grokking**, first discovered by Power et al.

The goal of the project was to reproduce Nanda's setup to build fluency with PyTorch, tensor manipulation and transformer internals. It then grew into four ablations (weight decay, two data fractions and a learning rate probe), each with predictions made beforehand so experimental hygiene could be practiced. An additional insight was gained when optimizer instability showed up (the slingshot effect) and got its own experiment.

The project does not attempt to reverse engineer the learned circuit - this was left out as it wasn't the intended goal.

## 2. Setup

The task given to the model was modular addition: predicting `(x + y) mod 113`. This is a standard grokking testbed from Power et al. and Neel Nanda's analysis. The input format is three tokens, x y =, the vocabulary is size 114 (0-112 plus the = token). The model predicts at the final position only, the loss is cross-entropy on that position and the other two positions are unsupervised. The dataset is the complete enumeration, all 113² = 12,769 combinations. Train and test are random partitions of that grid. (30% train = 3830 pairs at baseline). Every epoch is one full-batch pass over the entire train set.

The model is a 1-layer transformer. Learned token plus positional embeddings, one attention block (4 heads, width of each head is 32), one MLP (width 512, ReLU), linear unembed. The residual stream is 128 throughout, there is no LayerNorm and no dropout. Roughly 200k parameters.** The code was written from scratch in plain PyTorch by following the architecture of Nanda's reference but with the code rederived.** This was done to deepen understanding about transformer architecture and to gain tensor fluency.

### Training Config

| parameter | value |
|---|---|
| optimizer | AdamW |
| lr | 10⁻³ |
| weight decay | 1.0 |
| betas | (0.9, 0.98) |
| batch | full (all train pairs per step) |
| epochs | 50,000 (baseline) |
| frac_train | 0.3 (baseline) |
| hardware | Colab T4 |

The weight decay is so high precisely because we want to induce grokking, and a big regularization pressure does exactly that. The data split was seeded throughout, the model was unseeded.


## 3. Baseline result
![baseline loss curves](figures/baseline_curves.png)
*Baseline run (wd = 1.0, frac_train = 0.3). Train loss (blue) converges within a few hundred epochs; test loss (orange) generalizes ~10k epochs later — the delay is grokking. The recurring spikes are slingshot instabilities.*

### Grokking

The reason grokking happens in the first place is the following. Weight decay acts like a regularization force that penalizes large weights. The memorizing solution stores 3830 arbitrary answers and needs a large L2 norm to do it, while the generalizing circuit computes them with a much smaller norm. The weight decay pressure eventually favors the circuit. The hyperparameter of 30 percent training samples sits in the window where memorization is possible but costly.

The actual training follows a three act arc. Almost immediately in the first few hundred epochs the training loss collapses to 10⁻⁶ which means the model has memorized the training set. Meanwhile test loss rises above chance because the model's memorization machinery confidently produces wrong answers for the unseen pairs.

The model stays at this plateau of high test and low train loss for 10k epochs. It looks like nothing is happening but then suddenly...

...the cliff happens at around 10-12k. Test loss drops off by four orders of magnitude, the model suddenly generalizes. The delay between this and the train convergence is around 10k epochs. **Exactly this is grokking.** The delay exists because the model is silently building up the general circuit in parallel while memorization carries the loss. The cliff is the cleanup.

*My own prediction before running the experiment was that train would reach the lows at around 5k epochs and that test would reach it around 25k.* I was very conservative on both fronts, since train happened in the first few hundreds and test at around 10-12k.

### The slingshot effect

The slingshots are the recurring spikes, the loss jumping several orders of magnitude from the 10⁻⁷ floor up to 10⁻¹–10⁰. It recovers within a few hundred epochs and repeats quasi-periodically for the entire post-grok run.

This mechanism matches the "slingshot mechanism" identified in Thilak et al. It's a known instability of adaptive optimizers at very low loss. The reason is that Adam's normalization amplifies noise once gradients stay near zero long enough (mechanism in 4c).

Two key observations need to be noted. The first is that both curves spike and recover together. The second is that the model always falls back to the generalizing solution, never back to memorization.

An important fact is that this phenomenon occurred in every single configuration run. wd ∈ {0,1}, frac_train ∈ {0.1,0.3,0.5}, both learning rates and up to 100k epochs. **It's a constant in this setup.**

In conclusion it's an interesting unplanned second phenomenon that reproduced alongside the planned one.

## 4. Experimental variations

### 4a Weight decay = 0

![wd=0 loss curves](figures/wd0_curves.png)
*wd = 0, frac_train = 0.3. Test loss (orange) never generalizes — it climbs past 100 nats and keeps rising; train (blue) memorizes to 10⁻¹⁰; slingshot spikes persist without weight decay.*

*The predictions for this experiment were the following. The first was that there would be no grokking. The reason is that if wd is 0 there would be no pressure for the model to use the more efficient solution. For the train loss the prediction was that it would overfit even harder since nothing is opposing weight growth, in other words there is no regularization present. In the case of the slingshots the prediction was that they would persist since memorization alone would reach the near-zero-gradient regime where slingshots arise.*

The result confirmed all three. The test loss never dropped and it actually kept rising with the epochs. Train loss reached 10⁻¹⁰ vs the baseline's 10⁻⁶. The slingshot spikes also appeared as predicted.

The test loss is around 100 nats which means the model is very confidently wrong about its predictions. The chance loss would be around 4.7 nats, which means this model is twenty times worse than chance-level ignorance.

This experiment also shows that **wd is necessary for grokking, but not for slingshots**, the near-zero-gradient regime is reached either way.

There is a caveat worth mentioning, wd at 0 can show partial generalization via other implicit regularization but this wasn't observed here.

### 4b Fraction of training data = 0.5

![frac05 loss curves](figures/frac05.png)
*wd = 1.0, frac_train = 0.5. Test loss (orange) falls almost together with train (blue) — the ~10k delay of the baseline collapses to under ~1–2k epochs. Grokking nearly disappears when data is plentiful.*

*The prediction for this experiment was that the model groks earlier since it has two forces in the same direction. More pairs makes memorization more expensive and more training samples means the signal for finding the generalizing circuit is richer therefore it is found faster. The magnitude call was around 6-7k from the 10-12k baseline, naively inverse proportional to the increase of frac_train from 0.3 to 0.5, which is around 1.67x.*

The results confirm the direction of the prediction but expose that the magnitude was actually conservative by 5x. Test loss falls essentially with the train loss, **the delay between the losses is only about 1-2k which is very slim and it could arguably be said that no grokking is occurring, only phase change.**

This is in line with Power et al.'s findings where adding data shortens the wait for generalization disproportionately. The main takeaway is that **grokking is phase change and limited data combined.** At 0.5 that window is closing and the phase change is just training.

### 4c Learning rate drop (mechanism check on the slingshots)

![lr-drop loss curves](figures/lr-drop.png)
*Baseline resumed from epoch 20k with lr lowered 10× (10⁻³ → 10⁻⁴), wd unchanged. Spikes shrink by ~2 orders of magnitude but persist — a calm ~1.5k epochs precedes the first spike while the fresh optimizer state decays.*

The setup for the experiment was taking the baseline model that grokked and resuming it from its 20k epochs checkpoint and continuing its run until 50k epochs, but with a lowered learning rate, from 10⁻³ down to 10⁻⁴. A fresh optimizer was used, everything else was left as is.

This experiment checks the mechanism behind the slingshots. It is a known quirk of the Adam optimizer. 
Adam maintains two exponential moving averages, average of recent gradients - m (β₁ = 0.9 -> short ~ 10 step memory) and an average of recent squared gradients - v (β₂ = 0.98 -> ~50 step memory). The formula is essentially `step ∝ lr · m / (√v + ε)`, which means each param's raw gradient gets normalized by its own recent gradient scale.

This breaks because loss in this experiment reaches 10⁻⁷. The small loss causes small gradients. But they aren't just small, they are small for thousands of steps, and full batch training means no sampling noise to prop them up. As both m and v shrink together (because they are both relative to the size of the recent grads) their ratio stays in the order of 1. This is a problem because that means the step doesn't get smaller, it is roughly the size of the lr. And at a certain point instead of pointing in the direction of the gradient it points in the direction of the noise because Adam can't differentiate them due to magnitude.

The geometry of the kick is that the model sits at the bottom of an extremely flat and narrow basin (such a small loss 10⁻⁷ means the model is finely tuned) while the optimizer keeps making learning-rate-sized steps in noise-determined directions instead of gradient ones. Each step is small in the absolute weight space, but the basin floor is smaller than that. Eventually the random walk exits the basin, and because m is faster while v lags behind the exit gets amplified. For a few steps the denominator says the grad is small but the numerator says the grad is big and then the ratio spikes, the steps become huge, and the model gets slingshotted out of the basin. **That lag between m and v is the slingshot.**

The reason recovery is possible is that the gradients again get large and consistent. The ratio sanitizes and Adam starts working normally again and everything repeats.
The caveat is that ε at its default is too small here (10⁻⁸). A larger value would, in theory, raise the floor and prevent this from happening by reducing drifting and exiting the basin and by removing the amplification caused by lag.

*The prediction was that the spikes shrink roughly with the learning rate. The alternative was that they vanish entirely if the smaller kick can't escape.*

The results confirmed the prediction. **The peaks shrank but didn't vanish.** The amplitude was about 2 orders smaller when compared to the baseline spikes. From peaks of around 10⁻¹–10⁰ in the baseline to 10⁻³–10⁻² here. Since both runs start from the identical epoch 20k weights and differ only in the learning rate, the shrinkage therefore must be from the learning rate itself. 

Three interesting observations came out of it as well. The first was that the fresh Adam needed time to decay into the danger zone and was stable for about 1.5k epochs. The second was that the train loss was lowest in the calm around 10⁻⁷ and that the same number was never reached afterwards. This was because the learning rate is 10x lower and the optimizer doesn't have time to reach the basin's floor before the kick arrives again. The third was that test loss valleys actually get progressively lower, but still don't quite reach the first valley during the calm.

### 4d Fraction of training data = 0.1

![frac01 loss curves](figures/frac01at100k.png)
*wd = 1.0, frac_train = 0.1, run extended to 100k epochs. Test loss (orange) stays pinned at 30–70 nats throughout — no generalization, no downward trend. Train (blue) memorizes and slingshots indefinitely.*

*The prediction was that grokking could still occur but maybe past 50k epochs so the experiment was continued up to 100k epochs. The logic was that there would still be enough pressure to force the model to use the generalizing circuit, just that it would require more time.*

The results proved the prediction wrong. **The model never succeeded in generalization and it stayed stuck in memorization.** The test losses were pinned at around 30-70 nats through all the epochs with no downward trend at any point. The reason is that even though generalization should still be cheaper than memorization - the small amount of data needed to be memorized (1,276 pairs) doesn't exert enough pressure with the given weight decay - and therefore the model never switches to the generalization circuit since it doesn't see a need to. Power et al. got a similar kind of result. At this fraction of the training data the time to generalization grows steeply. **The conclusion we reach from this and 4b is that both too much and too little data can close the grokking window.**

## 5. Bug diary

**Loss frozen at 4.75** - loss was identical to 4 decimals every epoch. The reason was that the optimizer was created before the model was (re)instantiated, so it held parameter references to a dead object and stepped nothing. The optimizer binds at creation time, therefore recreate model = recreate optimizer.

**The causal mask wasn't applied** - the `scores.masked_fill(mask == 0, -1e10)` ran without error and did nothing. The reason was that `masked_fill` returns a new tensor rather than mutating and the result was never assigned. It was caught by printing the score matrix and noticing no -1e10 upper triangle.

**Plot crashed on loss lists** - the logging appended the raw loss tensors instead of .item() floats. The reason was that matplotlib refuses tensors that require grad.

## 6. What I didn't do

What this experiment didn't do was reverse-engineer the actual mechanism itself. Reproducing the Fourier analysis Nanda does was out of scope: the point was to learn more about the architecture, tensors and the grokking phenomenon.

The next steps are using the logit lens across the saved baseline checkpoints to inspect the general circuit forming during the plateau when the loss curves don't indicate anything is happening. The other thing that could be done is checking if a higher ε value floors the denominator and eliminates the amplification spikes entirely rather than shrinking them, as the theory suggests.


## 7. What I learned

The main goal was being more confident with tensor mechanics. I would say I succeeded, every shape was first predicted then checked and confirmed. Some tensor theory was also revisited, like the embedding lookup, broadcasting, softmax along an explicitly chosen axis.

More confidence was gained in regard to the transformer as well, especially the parameters, which matrix lives where, their shapes, and how the dimensions relate.

An unexpected thread where I learned something new was with slingshotting and Adam's internals, which demystified the optimizer a bit.

The last point is experimental hygiene. Every claim in section 4 rests on predictions written before the final results were in, which exercised my intuition and applied knowledge. Scoring them showed exactly where my mental model was off.


## References

1. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

2. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) — working reference for this project was the accompanying notebook, [A Mechanistic Interpretability Analysis of Grokking](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb)

3. Thilak, V., Littwin, E., Zhai, S., Saremi, O., Paiss, R., & Susskind, J. (2022). *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon.* [arXiv:2206.04817](https://arxiv.org/abs/2206.04817)