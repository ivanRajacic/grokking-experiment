Section 3 - grokking
![baseline loss curves](figures/baseline_curves.png)

Grokking

The baseline run has the following configuration of parameters that are known to produce grokking (grokking is a fragile phenomena). Weight decay is at 1.0, fraction of training samples used is 0.3 and we are running 50k epochs. Weight decay is the regularization force that makes sure the model picks the generalization circuit over the memorization one since it has a smaller weight l2 norm becuase its more efficient. 30 percent of the training data is enough for the model to get an understanding of the algorithm and big enough that the weight of memorization l2 norm is bigger than the generalization l2 weight norm. 50k epochs is to ensure that the model has enough time to actually perform the phase change and grok.

The actual training follows a three act arc. Almost immediatelly in the first few hundred epochs the training loss collapses to 10^-6 which means the model has memorized the training set. Meanwhile test loss rises above chance because the models memorization machinery confidently produces wrong answers for the unseen pairs.

The model stays at this plateu of high test and low train loss for 10k epochs. It looks like nothing is happening but then sudenlly...

The cliff happens at around 10-12k. Test loss drops off, the model suddenly generalizes. The delay between this and the train convergance is around 10k epochs. Exactly this is grokking. The delay exists because the model is silently building up the general circuit in parallel while memorization carries the loss. The cliff is the cleanup.

My own prediction before running the experiment was that train would reach the lows at around 5k epochs and that test would reach it around 25k. I was very conservative on both fronts, since train happend in the first few hundreds and test at around 10-12k.

Slingshots

The slingshots are the recurring spikes, the loss jumping several orders of magnitude from the 10^-7 floor up to 10^-1 - 10^0. It recovers within a few hundred epochs and repeats quasi periodically for the entire post-grok run.

This mechanism matches the "slingshot mechanism" identified in in Thilak et al. Its a known instability of adaptive optimizers at very low loss. The reason is that Adam's normalization amplifies noise once gradients stay near zero long enough.

Two key observations need to be noted. The first is that both curves spike and recover together. The second is that the model always falls back to the generalizing solution, never back to memorization.

An important fact is that this phenomena occured in every single configuration ran. wd e {0,1}, frac e {0.1,0.3,0.5}, both learning rates and up to 100k epochs. Its a constant in this setup.

In conlcusion its an intersting unplanned second phenomen that reproduced alongside the planned one.


Section 4 - experimental variations

4a Weight decay = 0

![baseline loss curves](figures/wd0_curves.png)

The predictions for this experiment were the following. The first was that there would be no grok happening. The reason is that because wd is 0 there would be no pressure for the model to use the more efficient solution. For the train loss the prediction was that it would overfit even harder since nothing is opposiong weight growth, in other words there is no regularisation present. In the case of the slingshots the prediction was that they would persist since memorization alone would reach Adam's low gradient habitat.

The result confirmed all three. The test loss never dropped and it actually kept rising with the epochs. Train loss reached 1e-10 vs the baseline's 1e-6. The slingshot spike also appeared as predicted.

The test loss is around 100 nats which means the model is very confidently wrong about its prediction. The average chance loss would be around 4.7 nats, which means this model is twenty times worse than chance ignorance.

This experiment also shows that wd is necessary for grokking but not for slingshots, the Adam's low gradient habitat is reached either way.

There is a caveat worth mentioning, wd at 0 can show partial generalization via other implicit regularization but this wasn't observed here.

4b Fraction of training data = 0.5

![baseline loss curves](figures/frac05.png)


The prediction for this experiment was that the model grokks earlier since it has two forces in the same direction. More pairs makes memorization more expensive and more training samples means the signal for finding the generalizing circuit is richer therefore it is found faster. The magnitude call was around 6-7k from the 10-12k baseline, proportional to the increase of frac_train from 0.3 to 0.5, around 1.67x.

The results confirm the direction of the prediction but expose that the magnitude was actually conservative by 5-10x. Test loss falls essentially with the train loss, the delay between the losses is only about 1-2k which is very slim and it could plausibly be said there is no grokking occuring, only phase change.

This is inline with Power et al's where data dependance is steeper than inverse-linear. The main takeway is that grokking is phase change and limited data combined. At 0.5 that window is closing and the phase change is just training.

4c Learning rate drop (mechanism check on slingshots)

![baseline loss curves](figures/lr-drop.png)


The setup for the experiment was taking the baseline model that grokked from the checkpoint 20k epochs continuing its run towards 50k but with a lowered learning rate, from 1e-3 down to 1e-4. Everything else is left as is.

This experiment checks the mechanism behind the slingshots. It is a known quirk of the Adam optimzer. 
Adam maintains two exponential moving averages, average of recent gradients - m (β₁ = 0.9 -> short ~ 10 step memory) and an average of recent squared gradients - v (β₂ = 0.98 -> ~50 step memory). The formula is essentially step ∝ lr · m / (√v + ε), which means each param's raw gradient gets normalized by its own recent gradient scale.
This breaks because loss in this experiment reaches 10^-7. The small loss causes small gradients. But they arent just small, they are small for thousands of steps, and full batch training means no sampling noise to prop them up. As both m and v shrink together (because they are both relative to the size of the recent grads) their ratio stays in the order of 1. This is a problem because that means the step doesn't get smaller, it roughly the size of the lr. And at a certain point instead of pointing in the direction of the gradient it points in the direction of the noise because Adam can't differentiate them due to magnitude.
The geomtry of the kick is that the models sits at the bottom of an extremly flat and narrow basin (such a small loss 10^-7 means the model is finely tuned) while the optimzer keeps making learning rate sized steps in noise determined directions instead of gradient ones. Each step is small in the absolute weight space, but the basin floor is smaller than that. Eventually the random walk exits the basin and because m is faster while v lags behind the exit gets amplified. For a few steps the denominator says the grad is small but the numerator says the grad is big and then the ratio spikes, the steps become huge, and the models gets slingshotted out of the basin. That lag between m and v is the slingshot.
The reason recovery is possible is that the gradients again get large and consistent. The ratio sanitizes and Adam starts working normally again and everything repeats.
The caveat is that ε at its default is too small here (10^-8). A larger value, would in theory, raise the floor and prevent this from happening by reducing drifitng and exiting the basin and by removing the amplification caused by lag.

The prediction was that the spikes shrink roughly with the learning rate. The alternative was that they vanish entirely if the smaller kick can't escape.

The results confirmed the prediction. The peaks shrunk but didn't vanish . The amplitude was about 2 orders smaller when compared to the baseline. From peaks of around 1e0 - 1e-1 in the baseline to 1e-3 - 1e-2 here. Since both runs start from the identical epoch 20k weights and differ only in the learning rate, the shrinkage therefore must be from the learning rate itself. 

Three interesting observations came out of it as well. The first was that the fresh Adam needed time to decay into the danger zone and was stable for about 1.5k epochs. The second was that the train loss was lowest in the calm around 1e-7 and that the same number was never reached afterwards. The third was that test loss valleys actually get progressively lower, but still dont quite reach the first valley during the calm.

4d Fraction of training data = 0.1

![baseline loss curves](figures/frac01at100k.png)

The prediction was that grokking could still occur but maybe past 50k epochs so the expriment was to continued up to 100k epochs. The logic was that there would still be enough pressure to force the model to use the generalizing circuit, just that it would require more time.

The results proved the prediction wrong. The model never succeded in generalization and it stayed stuck in the memorization. The test losses were pinned at around 30-70 nats through all the epochs with no downwards trend at any point. The reason is that even though generalization should still be cheaper than memorization the small amount of data needed to be memorized doesnt exert enough pressure with the given weight decay and therefore the model never switches to the generalization circuit since it doesnt see a need to. Power et al. got a similiar kind of result. At this fraction of the training data the time to generlization grows steeply. The conlusion we reach from this and 4b is that both too much and too little data can close the grokking window.



Missing — five sections, in the order you should write them:

Section 5 — Bug diary. The list, one line each, symptom → cause → lesson:

Optimizer created before the model it was meant to train → loss frozen at 4.75 → binding order matters; optimizer captures whatever model.parameters() points at now.
masked_fill computed and discarded (no assignment) → mask silently never applied → torch ops return, they don't mutate; check every bare expression line.
Same bug class in the MLP: forward line computed, never assigned → would-be silent architecture change; caught by shape luck.
Appended loss tensors instead of .item() → plotting crash + memory pinned by computation graphs.
x-axis built from num_epochs instead of from the logged data → two plotting crashes (fresh and resumed variants) → derive axes from data, never reconstruct.
Resume cell missing .to(device) → device-mismatch crash → both setup paths must set all run state (the five-things principle).
Fresh cell missing start_epoch = 0 → "fresh" runs silently resumed → same principle, other direction.
Run launched with the previous run's run_name → clobbered wd0's checkpoints; second folder incident mislabeled a figure → print run identity before the loop; makedirs + config print as unskippable cell code.

Section 2 — Setup. One paragraph: task ((x+y) mod 113, sequences x y =, loss on final position), architecture (1 layer, d_model 128, 4 heads, d_head 32, MLP 512, no LayerNorm, ~200k params, built from scratch — embedding/attention/MLP/unembed as separate modules). Then a hyperparameter table: lr 1e-3, wd 1.0, betas (0.9, 0.98), full-batch AdamW, frac_train 0.3, 50k epochs, T4/Colab. Seed footnote: split seeded throughout; init unseeded for baseline and wd0, torch.manual_seed thereafter — effects measured are orders of magnitude beyond seed variation. Link the notebook.

Section 6 — What I didn't do. One paragraph: the Fourier reverse-engineering — Nanda showed the grokked network computes (x+y) mod p via trigonometric identities on a few learned frequencies; reproducing that analysis was explicitly out of scope. Next steps, two lines: logit lens across the saved baseline checkpoints (watching the general circuit form during the plateau); the ε prediction — "the mechanism predicts a larger Adam ε floors the denominator and should eliminate the amplification phase of the spikes even where drift persists — untested here."

Section 7 — What I learned. Short and honest, three threads: tensor fluency (shapes derived before running — fancy indexing, broadcasting, batched matmul, the dim=-1 discipline); the init/scaling reasoning (√d everywhere and why); experiment hygiene (pre-registered predictions, run-named folders, checkpoint/resume, controls). One sentence on the meta-lesson if you want it: every bug in section 5 was invisible to shape checks — meaning-level tests (the ln(113) check, printing values) caught what types couldn't.

Section 1 — Intro, written last. Four-five sentences: grokking defined (memorize first, generalize abruptly much later); why it's interesting (cleanest toy case of sudden capability emergence); what this writeup is (from-scratch PyTorch reproduction + four ablations + an optimizer instability studied along the way); what it isn't (mechanism reverse-engineering — points at Nanda). Cite Power et al., Nanda, Thilak et al.

Plus the two closing chores: the end-pass (typo sweep via spellchecker, notation unified to 10⁻ˣ, e⁻¹⁰⁰ gloss in 4a, "Adam's low-gradient habitat" defined-or-replaced, "vanish ." space) and the overlay figure (test curves of frac 0.1/0.3/0.5 on one plot — one image, the whole data axis).

Write 5 → 2 → 6 → 7 → 1, end-pass last. Section 5 is pre-written above — transcribe, adjust voice, paste the rest as they land.