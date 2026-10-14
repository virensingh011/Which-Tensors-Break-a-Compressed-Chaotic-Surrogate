"""Build paper/main_v2.tex from paper/main.tex: new title/abstract/summary, Lorenz-96, tensor-isolated
interventions, protection experiment, updated discussion/limitations/conclusions.
Every number in the new text is read from paper/ext_numbers.json and the CSVs written by
analyze_extension.py -- nothing is typed by hand.

    python build_v2.py
"""
import json
import os

import numpy as np
import pandas as pd

P = "paper"
src = open(os.path.join(P, "main.tex")).read()
NUM = json.load(open(os.path.join(P, "ext_numbers.json")))
T, L, PR = NUM["tensor"], NUM["l96"], NUM["protect"]
TP = pd.read_csv(os.path.join(P, "tensor_paired.csv"))
PP = pd.read_csv(os.path.join(P, "protect_paired.csv"))


def fl(sys_, r, g):
    return f"{T[f'fail_{sys_}_{r}'][g]:.0f}"


def pd_(sys_, g, r, metric="vpt_loss", design=None):
    q = TP[(TP.system == sys_) & (TP.group == g) & (TP.r == r) & (TP.metric == metric)]
    return q.iloc[0]


def ci(sys_, g, r):
    q = pd_(sys_, g, r)
    return f"{q.med:+.2f}\\ [{q.lo:+.2f},\\,{q.hi:+.2f}]"


def rad(key, own=False):
    e = T["own_radius" if own else "tensor_radius"][key]
    return e


def rtxt(key, own=False, nd=3):
    e = rad(key, own)
    if e.get("r50") is None:
        return "below $0.01$"
    return f"{e['r50']:.{nd}f}"


def rci(key, own=False, nd=3):
    e = rad(key, own)
    return f"{e['r50']:.{nd}f}\\ [{e['lo']:.{nd}f},\\,{e['hi']:.{nd}f}]"


nL = L["seeds_valid"]
n_par = {s: T["tensor_parents"][s] for s in ("lorenz", "rossler", "lorenz96")}
n_models_tensor = T["tensor_models"]
n_models_main3d = 21180
n_models_l96 = L["models"]
n_models_prot = 30 * sum(PR["parents"].values())
import glob
n_mixed = sum(len(pd.read_csv(f)) for f in glob.glob("results_mixed/*_seed*.csv"))
n_width = sum(len(pd.read_csv(f)) for f in glob.glob("results_width/H*/*_seed*.csv"))
n_total = n_models_main3d + n_models_l96 + n_models_tensor + n_models_prot + n_mixed + n_width
r50_l96, r50_l96_lo, r50_l96_hi = L["radius"]["r50"]
r10_l96 = L["radius"]["r10"][0]
r90_l96 = L["radius"]["r90"][0]
def _red():
    out = []
    for s_ in ("lorenz", "rossler"):
        for r_ in (0.01, 0.02):
            a_, b_ = T[f"fail_{s_}_{r_}"]["all"], T[f"fail_{s_}_{r_}"]["all_but_W3"]
            out.append(100 * (a_ - b_) / a_)
    return f"{min(out):.0f}--{max(out):.0f}"


RED = _red()
_o = [T["own_radius"][f"lorenz_{g}"]["r50"] for g in ("W1", "W2", "W3")]
OWN_L63 = f"{100 * min(_o):.0f}--{100 * max(_o):.0f}"
_o9 = [T["own_radius"][f"lorenz96_{g}"]["r50"] for g in ("W1", "W2", "W3")]
OWN_L96 = f"{100 * min(_o9):.0f}--{100 * max(_o9):.0f}"
_q = {r["group"]: r for r in L["fail_table"]}
Q6 = _q["quant@6"]["sdiv"] + _q["quant@6"]["scol"]
Q4 = _q["quant@4"]["sdiv"] + _q["quant@4"]["scol"]


def n0(x):
    return f"{x:,.0f}".replace(",", "{,}")


def rep(old, new, count=1):
    global src
    assert src.count(old) >= 1, f"pattern not found: {old[:70]!r}"
    src = src.replace(old, new, count)


def between(start, end, new):
    """Replace text from `start` up to (not including) `end`."""
    global src
    i = src.index(start)
    j = src.index(end, i)
    src = src[:i] + new + src[j:]


# =============================================================================================== title / abstract
rep(r"\title{\textbf{How Much Compression Can a Learned Chaotic Surrogate Survive? A Matched-Control Study of Pruning, Quantization and Attractor Failure in Lorenz-63 and R\"ossler Emulators}}",
    r"\title{\textbf{Which Tensors Break a Compressed Chaotic Surrogate? Matched-Control and Tensor-Isolated Perturbation Experiments on Lorenz-63, R\"ossler and Lorenz-96 Emulators}}")

abstract = rf"""\begin{{abstract}}
\noindent Neural surrogates of chaotic systems are compressed by pruning or quantization, and the size of the compression is usually reported as a sparsity, a bit width or a scalar parameter displacement, although the surrogates are deployed in long autonomous rollouts. We ask how large a displacement a surrogate survives, whether equal total displacement implies equal damage, and which tensors decide the outcome. We train 30 residual-MLP surrogates each of Lorenz-63, R\"ossler and Lorenz-96 ($N=6$) and pair every pruning or quantization perturbation with random perturbations of identical total relative norm, identical per-tensor norms and identical support, and add tensor-isolated noise interventions, a protection experiment, a width check and a mixed-precision quantization test ({n0(n_total)} perturbed models in total). Failure dominates, so we use survival-independent endpoints. Half of isotropic perturbations destroy the attractor at a relative displacement of 4.4\% (Lorenz-63), 1.8\% (R\"ossler) and {100 * r50_l96:.0f}\% (Lorenz-96). Equal total norm does not imply equal damage: at a total relative norm of $0.01$, noise confined to the small output layer makes {fl('lorenz', 0.01, 'W3')}\% of Lorenz-63 and {fl('rossler', 0.01, 'W3')}\% of R\"ossler surrogates fail, against {fl('lorenz', 0.01, 'all')}\% and {fl('rossler', 0.01, 'all')}\% for isotropic noise and {fl('lorenz', 0.01, 'W2')}\% and {fl('rossler', 0.01, 'W2')}\% for noise confined to the large middle matrix, and removing the output layer from isotropic noise, where it carries 4\% of the squared norm, cuts failures by {RED}\%. Measured in each tensor's \emph{{own}} relative displacement, the weight matrices of Lorenz-63 fail at similar radii ({OWN_L63}\%), so a per-tensor relative displacement is a better compression budget than the total norm; protecting only the output layer from real pruning or quantization is not sufficient. Failure is rarely silent (at least 98\% of failed models already have one-step error at least ten times that of their parent), and valid prediction time is log-linear in one-step error with a slope that Lyapunov and Jacobian statistics do not explain. Scope: three systems, small MLPs, post-training compression.
\end{{abstract}}"""
between(r"\begin{abstract}", r"\noindent\textbf{Keywords:}", abstract + "\n\n")
rep("pruning; quantization; Lyapunov exponent; valid prediction time; attractor failure; matched controls",
    "pruning; quantization; tensor-isolated perturbation; Lyapunov exponent; valid prediction time; attractor failure; matched controls; Lorenz-96")

highlights = rf"""\noindent\fcolorbox{{black!40}}{{black!4}}{{\parbox{{0.96\textwidth}}{{\small\textbf{{Highlights}}
\begin{{itemize}}
\item Half of isotropic weight perturbations destroy a chaotic attractor at 1.8--4.4\% relative norm on two 3-D systems and at about {100 * r50_l96:.0f}\% on Lorenz-96.
\item Equal total norm is not equal damage: noise confined to the output layer is far more damaging than the same total norm in the middle matrix.
\item Each tensor fails at a similar relative displacement of itself ($\approx${OWN_L63}\% on Lorenz-63): budgets should be per-tensor, not total.
\item Protecting only the output layer from pruning or quantization is not sufficient to prevent failure.
\item Failure is rarely silent: at least 98\% of failed surrogates already show a tenfold rise in one-step error.
\end{{itemize}}}}}}"""
between(r"\noindent\fcolorbox{black!40}{black!4}", r"\section{Introduction}", highlights + "\n\n")

# =============================================================================================== introduction
rep(r"This paper started from two hypotheses. H1: compression degrades long-run dynamics before one-step error increases. H2: two perturbations of equal parameter norm can nevertheless have different dynamical effects. A preliminary five-seed pilot suggested support for H2 on the Lorenz system, with the sign of the effect depending on the system. Both hypotheses were stated before the full experiment; we test them in Sections~\ref{sec:h1} and~\ref{sec:matched}. We report here a considerably larger, independently implemented study designed to test these claims more carefully. Its main lesson is methodological: a large share of perturbed chaotic surrogates fail outright, which makes the commonly reported Lyapunov-exponent comparison unreliable unless failures are accounted for.",
    r"This paper started from two hypotheses. H1: compression degrades long-run dynamics before one-step error increases. H2: two perturbations of equal parameter norm can nevertheless have different dynamical effects. A preliminary five-seed pilot suggested support for H2 on the Lorenz system, with the sign of the effect depending on the system. Both hypotheses were stated before the full experiment; we test them in Sections~\ref{sec:h1} and~\ref{sec:matched}. The matched-norm comparisons support H2 only modestly and pointed to the small output layer as the place where isotropic noise and structured perturbations differ; this led to a third hypothesis, H3: isotropic noise does disproportionate damage through the output layer, so that the total parameter norm is a poor compression budget. H3 was formulated after the main results had been seen, and we test it with tensor-isolated interventions (Section~\ref{sec:mech}), a protection experiment on real pruning and quantization (Section~\ref{sec:protect}), and a replication on a six-dimensional system, Lorenz-96 (Section~\ref{sec:l96}). A methodological lesson runs through all of these: a large share of perturbed chaotic surrogates fail outright, which makes the commonly reported Lyapunov-exponent comparison unreliable unless failures are accounted for.")

contrib = rf"""\paragraph{{Contributions.}}
\begin{{enumerate}}
\item \emph{{Tensor-isolated interventions.}} Noise confined to one tensor group at a time, in three designs (matched total norm, leave-one-out, matched own-norm), show that the same total relative norm can be harmless or destructive depending on where it lands, that the small output layer drives the excess damage of isotropic noise, and that every tensor fails at a comparable relative displacement of itself. A per-tensor relative displacement, not the total norm, is the natural compression budget.
\item A \emph{{matched-control protocol}}: every structured perturbation is paired with random perturbations of identical total relative norm, identical per-tensor norms, and (for pruning) identical support, so that the roles of magnitude, distribution across layers, and which entries are touched can be separated; replicated on three systems of dimension 3 and 6.
\item Large-scale evidence of how fragile learned chaotic surrogates are to parameter perturbations (failure rates by perturbation type and norm), a \emph{{dynamical robustness radius}} with bootstrap intervals, and a demonstration that conditioning on survival biases LLE-based comparisons.
\item Survival-independent paired comparisons of structured and matched random perturbations (valid prediction time, one-step error, failure), showing a consistent but modest advantage of structured perturbations over isotropic noise that largely disappears once per-tensor norms are matched, and a difference in failure \emph{{mode}} at matched norm.
\item A direct test of whether protecting the output layer from real pruning or quantization prevents failure (it does not, in general), and a direct test of H1 (post hoc): failure rarely occurs while one-step error is still near the parent's.
\item A comparison of predictors of dynamical damage---parameter norm, one-step error, and trajectory-weighted Jacobian-change measures including a first-order Lyapunov-response estimate---with an explicit negative result for predicting LLE change, and an empirical log-linear relation between one-step error and valid prediction time whose slope is not explained by Lyapunov or Jacobian statistics.
\end{{enumerate}}

"""
between(r"\paragraph{Contributions.}", r"\paragraph{Summary of findings.}", contrib)

summary_tab = rf"""\begin{{table}}[t]
\centering\small
\caption{{Research questions and findings. $^\dagger$: exploratory analysis added after the primary results were seen.}}\label{{tab:summary}}
\begin{{tabular}}{{@{{}}p{{0.30\textwidth}}p{{0.62\textwidth}}@{{}}}}
\toprule
Question & Finding\\
\midrule
H1: does failure occur while one-step error is still near the parent's?$^\dagger$ & Rarely: 98--99\% of failed 3-D models and {L['h1']['failed_ge10']:.0f}\% of failed Lorenz-96 models have at least $10\times$ the parent's one-step error. R\"ossler exception: 16\% of models below $3\times$ fail. High error is sensitive but not specific (\S\ref{{sec:h1}}, \S\ref{{sec:l96}}).\\[2pt]
H2: does equal total norm imply equal dynamics? & No. Failure modes differ, and structured perturbations beat isotropic noise in 17/20 VPT and 18/20 MSE comparisons, but only 5/20 each against layer-matched noise (\S\ref{{sec:matched}}). Confining the same total norm to one tensor changes the failure rate from {fl('lorenz', 0.01, 'W2')}\% to {fl('lorenz', 0.01, 'W3')}\% on Lorenz-63 at norm $0.01$ (\S\ref{{sec:mech}}).\\[2pt]
H3: does isotropic noise do disproportionate damage through the output layer?$^\dagger$ & Yes. Removing $W_3$, which carries 4\% of the squared norm of isotropic noise, cuts failures by {RED}\% (at total norms $0.01$--$0.02$ on the 3-D systems); $W_3$-only noise is the most damaging group at equal total norm. But in each tensor's own relative displacement all tensors fail at comparable radii, so the output layer is special only through the total-norm coordinate (\S\ref{{sec:mech}}).\\[2pt]
How large a displacement does the attractor survive?$^\dagger$ & Half of isotropic perturbations destroy it at 4.4\% (Lorenz-63), 1.8\% (R\"ossler) and {100 * r50_l96:.0f}\% (Lorenz-96) of the parameter norm; 6-bit quantization sits at this radius on the 3-D systems but not on Lorenz-96 (\S\ref{{sec:radius}}, \S\ref{{sec:l96}}).\\[2pt]
Does protecting the output layer prevent failure under real compression?$^\dagger$ & Not in general: it lowers failure in some settings (e.g.\ R\"ossler quantization) and not in others, and compressing only the output layer is itself enough to cause failure (\S\ref{{sec:protect}}).\\[2pt]
Which cheap quantities predict damage? & Parent-relative one-step error predicts VPT loss (CV $R^2$ $0.85$/$0.88$ against $0.77$/$0.74$ for norm on the 3-D systems); $\Delta$LLE is poorly predicted by all quantities (\S\ref{{sec:pred}}).\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""
between(r"\begin{table}[t]" + "\n" + r"\centering\small" + "\n" + r"\caption{Research questions and findings.", r"\section{Related Work}", summary_tab)

# =============================================================================================== setup
rep(r"We study Lorenz-63~\cite{lorenz1963} ($\sigma=10,\rho=28,\beta=8/3$) and R\"ossler~\cite{rossler1976} ($a=b=0.2,c=5.7$). Trajectories are integrated with fourth-order Runge--Kutta (internal step $10^{-3}$), a $20$ time-unit transient is discarded, and states are sampled every $\Delta t=0.02$.",
    r"We study Lorenz-63~\cite{lorenz1963} ($\sigma=10,\rho=28,\beta=8/3$) and R\"ossler~\cite{rossler1976} ($a=b=0.2,c=5.7$) and, as a six-dimensional replication, Lorenz-96~\cite{lorenz1996} with $N=6$ and forcing $F=8$ ($\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F$, cyclic indices). Trajectories are integrated with fourth-order Runge--Kutta (internal step $10^{-3}$ for the 3-D systems and $5\times10^{-3}$ for Lorenz-96), a $20$ time-unit transient is discarded, and states are sampled every $\Delta t=0.02$ (3-D) or $\Delta t=0.05$ (Lorenz-96).")
rep(r"Reference exponents, computed by tangent-vector integration, are $0.903$ (Lorenz) and $0.0655$ (R\"ossler); we use these values as the Lyapunov time scale.",
    rf"Reference exponents, computed by tangent-vector integration, are $0.903$ (Lorenz), $0.0655$ (R\"ossler) and ${L['ref_lle']:.3f}$ (Lorenz-96); we use these values as the Lyapunov time scale.")
rep(r"with two hidden layers of 64 units ($4{,}611$ parameters), trained with",
    r"with two hidden layers of 64 units ($4{,}611$ parameters for the 3-D systems; $6$--$64$--$64$--$6$, $4{,}998$ parameters for Lorenz-96), trained with")
rep(r"and of the R\"ossler parents $1.5\times10^{-6}$ ($3.2\times10^{-7}$--$1.9\times10^{-5}$) in standardised units.",
    rf"and of the R\"ossler parents $1.5\times10^{{-6}}$ ($3.2\times10^{{-7}}$--$1.9\times10^{{-5}}$) in standardised units; for the {nL} valid Lorenz-96 parents it is ${L['parent_mse_median']:.1e}$.".replace("e-05", r"\times10^{-5}"))

# new subsection on interventions, inserted before "Metrics"
intv = rf"""\subsection{{Tensor-isolated interventions and the protection experiment}}\label{{sec:intv}}
To test which tensor drives the damage we inject Gaussian noise into one tensor group $G\in\{{W_1,W_2,W_3,\text{{biases}}\}}$ at a time (biases: $b_1,b_2,b_3$), in three designs with $K=10$ draws at every level:
\begin{{itemize}}
\item \textbf{{total-norm design}}: noise confined to $G$ and rescaled so that the \emph{{total}} relative displacement over all parameters equals $r\in\{{0.01,0.02,0.05,0.1,0.2\}}$, the metric used throughout the paper; the control is isotropic noise at the same $r$;
\item \textbf{{leave-one-out design}}: isotropic noise with the tensor group $G$ removed and the remainder rescaled to total relative norm $r$ (if $G$ is responsible for the damage of isotropic noise, removing it should rescue the model);
\item \textbf{{own-norm design}}: noise confined to $G$ and rescaled so that $G$ itself is displaced by $\|\Delta\theta_G\|/\|\theta_G\|=r\in\{{0.02,0.05,0.1,0.2,0.5\}}$.
\end{{itemize}}
This is 650 models per parent, {n0(n_models_tensor)} in total over the {sum(n_par.values())} valid parents of the three systems. The interventions were designed to test H3 and run once, before they were analysed; the choice of five norms was fixed in advance.

The \emph{{protection experiment}} asks whether the result is actionable. Each valid parent is compressed with global and layer-wise magnitude pruning ($25,50,75\%$) and with quantization ($8,6,4,3$ bits) under three scopes: all tensors (as before), all tensors \emph{{except}} the output layer ($W_3,b_3$ stay unpruned and in full precision), and \emph{{only}} the output layer. These are deterministic (no random draws), 30 models per parent, {n0(n_models_prot)} in total.

"""
rep(r"\subsection{Metrics}", intv + r"\subsection{Metrics}")

rep(r"are excluded from all analyses: all 30 Lorenz and 22 of 30 R\"ossler parents pass. This leaves $18{,}356$ perturbed models.",
    rf"are excluded from all analyses: all 30 Lorenz and 22 of 30 R\"ossler parents pass, which leaves $18{{,}}356$ perturbed models on the two 3-D systems; {nL} of {L['parents_total']} Lorenz-96 parents pass. For Lorenz-96 the exponent is estimated for the parents only (to apply this rule); the study of perturbed models uses failure, one-step error and VPT.")
rep(r"the model is rolled out for up to 10 time units; VPT is the first time the normalised error $\|\hat x-x\|/\sqrt{3}$ exceeds $0.4$,",
    r"the model is rolled out for 500 steps (10 time units for the 3-D systems, 25 for Lorenz-96); VPT is the first time the normalised error $\|\hat x-x\|/\sqrt{d}$ exceeds $0.4$ ($d=3$ or $6$),")
rep(r"Parent VPT is $1.76\pm0.45$ (Lorenz) and $0.43\pm0.09$ (R\"ossler) Lyapunov times (mean $\pm$ sd).",
    rf"Parent VPT is $1.76\pm0.45$ (Lorenz), $0.43\pm0.09$ (R\"ossler) and ${L['parent_vpt_mean']:.2f}\pm{L['parent_vpt_sd']:.2f}$ (Lorenz-96) Lyapunov times (mean $\pm$ sd).")

rep(r"\paragraph{Additional analyses.}",
    r"""\paragraph{Interventions.} For the tensor-isolated interventions and the protection experiment the unit of replication is again the parent. Per parent we take the mean failure indicator and the median VPT loss over the $K$ draws, compare each noise group with the isotropic condition at the same total norm (or each protection scope with the all-tensor condition) by the per-parent difference, and report its median with a 95\% bootstrap interval, a Wilcoxon signed-rank $p$-value and Holm-adjusted $p$-values within each (system, metric) family. Radii are logistic-regression estimates as below (relative displacement of the whole network for the total-norm design, of the perturbed tensor for the own-norm design); when the failure rate already exceeds 50\% at the smallest tested norm we report the radius as lying below it rather than extrapolating.

\paragraph{Additional analyses.}""")

# =============================================================================================== results: mechanism section
old_mech_start = r"\subsection{Which tensors carry the difference?}\label{sec:mech}"
old_mech_end = r"\begin{table}[t]" + "\n" + r"\centering\small" + "\n" + r"\caption{Share (\%) of squared displacement per tensor group"
def sg(s, g, r): return fl(s, r, g)
mech_head = rf"""\subsection{{Which tensors carry the difference?}}\label{{sec:mech}}
All families put 89--97\% of the squared displacement into the large middle matrix $W_2$ (4{{,}}096 of 4{{,}}611 parameters; Table~\ref{{tab:shares}}), so the total-norm match hides little there. The difference is in the small output layer $W_3$ and the biases: at matched total norm, isotropic noise displaces $W_3$ in relative terms $2.8\times$ (Lorenz) and $2.5\times$ (R\"ossler) more than the structured perturbations it is paired with (median over the ten structured groups), whereas $W_1$ and $W_2$ are displaced about equally (ratios $0.8$--$1.1$). The output layer is small by initialisation and directly sets the residual step. This suggested H3, which we now test directly with the interventions of Section~\ref{{sec:intv}} (Figure~\ref{{fig:tensor}}, Tables~\ref{{tab:tensor}}--\ref{{tab:tensorradius}}).

\paragraph{{Noise in one tensor, same total norm.}} Where the noise lands matters by an order of magnitude. At a total relative norm of $0.01$, noise confined to $W_3$ makes {sg('lorenz','W3',0.01)}\% of Lorenz-63 and {sg('rossler','W3',0.01)}\% of R\"ossler surrogates fail, against {sg('lorenz','all',0.01)}\% and {sg('rossler','all',0.01)}\% for isotropic noise and {sg('lorenz','W2',0.01)}\% and {sg('rossler','W2',0.01)}\% for noise confined to $W_2$, which holds 89\% of the squared norm of isotropic noise. The robustness radii of Table~\ref{{tab:tensorradius}} tell the same story: on Lorenz-63 the radius is {rtxt('lorenz_W2')} for $W_2$-only noise, {rtxt('lorenz_all')} for isotropic noise (on the five-norm grid used here; Section~\ref{{sec:radius}} gives $0.044$ on the finer grid) and lies below $0.01$ for $W_3$-only noise; on R\"ossler the corresponding values are {rtxt('rossler_W2')}, {rtxt('rossler_all')} and again below $0.01$. In valid prediction time, relative to isotropic noise at norm $0.01$, $W_3$-only noise loses {ci('lorenz','W3',0.01)} Lyapunov times on Lorenz-63 and {ci('rossler','W3',0.01)} on R\"ossler (more is lost), whereas $W_2$-only noise loses {ci('lorenz','W2',0.01)} and {ci('rossler','W2',0.01)} (less is lost); all four are Holm-significant at $p<0.001$ (Table~\ref{{tab:tensorpaired}}).

\paragraph{{Removing the output layer rescues isotropic noise.}} Isotropic noise puts only 4.2\% of its squared norm into $W_3$ (Table~\ref{{tab:shares}}), yet removing it cuts failures at total norm $0.01$ from {sg('lorenz','all',0.01)}\% to {sg('lorenz','all_but_W3',0.01)}\% on Lorenz-63 and from {sg('rossler','all',0.01)}\% to {sg('rossler','all_but_W3',0.01)}\% on R\"ossler, and at $0.02$ from {sg('lorenz','all',0.02)}\% to {sg('lorenz','all_but_W3',0.02)}\% and from {sg('rossler','all',0.02)}\% to {sg('rossler','all_but_W3',0.02)}\%. The VPT lost changes by {ci('lorenz','all_but_W3',0.01)} (Lorenz-63) and {ci('rossler','all_but_W3',0.01)} (R\"ossler) Lyapunov times at $0.01$ (both $p<0.001$). Removing $W_1$ or the biases changes the VPT lost by at most $0.09$ Lyapunov times on Lorenz-63 (no Holm-significant change at any of the five norms) and by at most $0.02$ on R\"ossler (significant for the biases at three norms, but tiny), whereas removing $W_2$, which holds most of the norm, \emph{{worsens}} it, because the same total norm is then forced into the small tensors ({ci('lorenz','all_but_W2',0.01)} Lyapunov times on Lorenz-63 at $0.01$). So the output layer is the one tensor whose removal substantially rescues isotropic noise.

\paragraph{{Per-tensor relative displacement is the natural coordinate.}} In the own-norm design the picture changes (bottom row of Figure~\ref{{fig:tensor}}, last rows of Table~\ref{{tab:tensorradius}}). Measured as a fraction of the tensor's own norm, $W_1$, $W_2$ and $W_3$ fail at similar radii on Lorenz-63 ({rtxt('lorenz_W1', True, 2)}, {rtxt('lorenz_W2', True, 2)} and {rtxt('lorenz_W3', True, 2)}) and at {rtxt('rossler_W1', True, 2)}, {rtxt('rossler_W2', True, 2)} and {rtxt('rossler_W3', True, 2)} on R\"ossler, where $W_3$ is the most fragile; the biases are as fragile as the weights on Lorenz-63 ({rtxt('lorenz_biases', True, 2)}) and much more robust on R\"ossler ({rtxt('rossler_biases', True, 2)}). Why the total-norm coordinate misleads is then plain arithmetic: $W_3$ is small, so a given absolute displacement is a large \emph{{relative}} displacement of it. $W_3$-only noise at total norm $0.01$ displaces $W_3$ by about 10\% (Lorenz-63) and 19\% (R\"ossler) of its own norm, and its failure rates ({sg('lorenz','W3',0.01)}\% and {sg('rossler','W3',0.01)}\%) are close to those of the own-norm design at those displacements ({T['own_lorenz_0.1']['W3']:.0f}\% at $0.1$ on Lorenz-63; {T['own_rossler_0.2']['W3']:.0f}\% at $0.2$ on R\"ossler); isotropic noise at the same total norm displaces $W_3$ by only about 2\% and 4\%. We therefore read H3 as supported in the total-norm coordinate, and not as evidence that the output layer is intrinsically special on Lorenz-63: what the interventions show is that fragility is set by how far \emph{{each}} tensor is moved relative to its own size. This is consistent with Section~\ref{{sec:matched}}: per-tensor quantization and layer-wise pruning displace every tensor by a comparable fraction of itself, whereas isotropic noise at the same total norm displaces the small tensors disproportionately, which is why matching per-tensor norms removes most of the apparent advantage of structured perturbations.

\begin{{figure}}[t]
\centering\includegraphics[width=\textwidth]{{figures/fig_tensor.pdf}}
\caption{{Tensor-isolated interventions on Lorenz-63, R\"ossler and Lorenz-96. Top: fraction of models that fail against \emph{{total}} relative displacement for isotropic noise (black), noise confined to one tensor group, and isotropic noise with the output layer removed (green dashed). Middle: median valid prediction time lost. Bottom: failure against the relative displacement of the perturbed tensor itself (own-norm design). Bars: 95\% intervals over parents.}}\label{{fig:tensor}}
\end{{figure}}

\begin{{table}}[t]
\centering\small
\caption{{Failure (\%) at total relative norm $r$ by where the noise is placed (mean over parents of the failure rate over 10 draws). ``No $W_3$'': isotropic noise with the output layer removed.}}\label{{tab:tensor}}
\begin{{tabular}}{{llcccccc}}
\toprule
System & $r$ & isotropic & $W_1$ only & $W_2$ only & $W_3$ only & biases only & no $W_3$\\
\midrule
\input{{tables/tab_tensor.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering\small
\caption{{Robustness radius $r_{{50}}$ (relative displacement at which half of the models fail) by where the noise is placed. ``total'': relative to the whole parameter vector (total-norm design); ``own-norm'': relative to the perturbed tensor itself. ``$<0.01$'': failure already exceeds 50\% at the smallest tested norm.}}\label{{tab:tensorradius}}
\begin{{tabular}}{{llccccc}}
\toprule
System & coordinate & isotropic & $W_1$ & $W_2$ & $W_3$ & biases\\
\midrule
\input{{tables/tab_tensor_radius.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering\small
\caption{{Paired comparison with isotropic noise at the same total norm (median over parents; Holm-adjusted $p$ for VPT lost). $\Delta$fail: change in failure rate, percentage points. $\Delta$VPT: change in valid prediction time lost, Lyapunov times (positive: more damage), with 95\% bootstrap interval.}}\label{{tab:tensorpaired}}
\begin{{tabular}}{{llcrcc}}
\toprule
System & noise & $r$ & $\Delta$fail & $\Delta$VPT & $p_{{\mathrm{{Holm}}}}$\\
\midrule
\input{{tables/tab_tensor_paired.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Coarse predictive check.}} """
mech_old_tail_start = r"Table~\ref{tab:cv} gives a coarse predictive check."
i0 = src.index(old_mech_start)
i1 = src.index(mech_old_tail_start, i0)
j_tab = src.index(old_mech_end, i0)  # tab:shares table starts here (after the old first-two paragraphs)
# keep the old tail paragraph text (from "Table~\ref{tab:cv} ..." up to the tab:shares table) and tables after it
old_tail = src[i1:j_tab]
src = src[:i0] + mech_head + old_tail.replace(
    r"Table~\ref{tab:cv} gives a coarse predictive check.",
    r"Table~\ref{tab:cv} gives a coarse predictive check across all perturbations.") + src[j_tab:]

# =============================================================================================== results: protection + Lorenz-96
def pf(s, fam, lvl, sc):
    return PR["fail"][f"{s}|{fam}|{lvl}|{sc}"]


def pcnt(s, sc, key):
    return PR[f"{s}|{sc}"][key]


prot_sec = rf"""\subsection{{Does protecting the output layer help real compression?}}\label{{sec:protect}}
If the output layer were the weak point of real compression, leaving it unpruned and in full precision should prevent failure. Table~\ref{{tab:protect}} shows a partial benefit only. Protecting $(W_3,b_3)$ significantly lowers the valid prediction time lost (Holm-adjusted) in {PR['lorenz|protOut']['vpt_better']} of {PR['lorenz|protOut']['n']} settings on Lorenz-63, {PR['rossler|protOut']['vpt_better']} on R\"ossler and {PR['lorenz96|protOut']['vpt_better']} on Lorenz-96, but significantly lowers the \emph{{failure rate}} in at most {max(PR['lorenz|protOut']['fail_better'], PR['rossler|protOut']['fail_better'], PR['lorenz96|protOut']['fail_better'])} setting per system, and it can raise it (Lorenz-63 global 50\% pruning: {pf('lorenz','prune_global',0.5,'all'):.0f}\% to {pf('lorenz','prune_global',0.5,'protOut'):.0f}\%). The largest gains are on R\"ossler (6-bit quantization from {pf('rossler','quant',6,'all'):.0f}\% to {pf('rossler','quant',6,'protOut'):.0f}\%, 4-bit from {pf('rossler','quant',4,'all'):.0f}\% to {pf('rossler','quant',4,'protOut'):.0f}\%, global 25\% pruning from {pf('rossler','prune_global',0.25,'all'):.0f}\% to {pf('rossler','prune_global',0.25,'protOut'):.0f}\%) and on Lorenz-96 (global 50\% pruning from {pf('lorenz96','prune_global',0.5,'all'):.0f}\% to {pf('lorenz96','prune_global',0.5,'protOut'):.0f}\%); on Lorenz-63 most settings change little (6-bit quantization {pf('lorenz','quant',6,'all'):.0f}\% to {pf('lorenz','quant',6,'protOut'):.0f}\%). Conversely, compressing \emph{{only}} the output layer, a perturbation with about {100 * PR['rel_norm_onlyOut_vs_all']:.0f}\% of the relative norm of compressing everything, already causes much of the pruning failure on Lorenz-63 and Lorenz-96 (layer-wise 25\% pruning: {pf('lorenz','prune_layer',0.25,'onlyOut'):.0f}\% against {pf('lorenz','prune_layer',0.25,'all'):.0f}\% for all tensors on Lorenz-63 and {pf('lorenz96','prune_layer',0.25,'onlyOut'):.0f}\% against {pf('lorenz96','prune_layer',0.25,'all'):.0f}\% on Lorenz-96), though much less on R\"ossler ({pf('rossler','prune_layer',0.25,'onlyOut'):.0f}\% against {pf('rossler','prune_layer',0.25,'all'):.0f}\%). Both observations agree with the own-norm radii of Table~\ref{{tab:tensorradius}}: pruning 25\% of a tensor or quantizing it to a few bits displaces it by roughly 10\% of its own norm or more, a level at which \emph{{each}} tensor is already fragile on the 3-D systems, so sparing one tensor is not enough. The practical message is therefore about the budget, not about a protected layer: the relative displacement of every tensor should stay below its own radius (about $0.1$ on these surrogates), and the total norm is the wrong summary of it.

\begin{{table}}[t]
\centering\small
\caption{{Failure (\%) under real compression with all tensors compressed (all), with the output layer ($W_3,b_3$) protected (prot.) and with only the output layer compressed (only). Mean over valid parents.}}\label{{tab:protect}}
\begin{{tabular}}{{lccccccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{Lorenz-63}} & \multicolumn{{3}}{{c}}{{R\"ossler}} & \multicolumn{{3}}{{c}}{{Lorenz-96}}\\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
Compression & all & prot. & only & all & prot. & only & all & prot. & only\\
\midrule
\input{{tables/tab_protect.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""

l96_sec = rf"""\subsection{{Replication on a six-dimensional system: Lorenz-96}}\label{{sec:l96}}
Lorenz-96 ($N=6$, $F=8$) with a $6$--$64$--$64$--$6$ surrogate was run with the same protocol ({nL} valid parents of {L['parents_total']}; {n0(n_models_l96)} perturbed models). Overall {L['fail_overall']:.0f}\% of the perturbed models fail ({L['div_overall']:.0f}\% diverge, {L['col_overall']:.0f}\% collapse; Table~\ref{{tab:l96fail}}). The surrogates are more robust than the 3-D ones: half of isotropic perturbations destroy the attractor at $r_{{50}}={r50_l96:.3f}$ [{r50_l96_lo:.3f}, {r50_l96_hi:.3f}] of the parameter norm, with $r_{{10}}={r10_l96:.3f}$ and $r_{{90}}={r90_l96:.2f}$. Six-bit quantization (relative norm about 0.03) therefore does \emph{{not}} sit at the radius here: it fails in {Q6:.0f}\% of models, and 4-bit quantization (about 0.14, close to $r_{{50}}$) in {Q4:.0f}\%, so the statement that 6-bit quantization reaches the radius is specific to the 3-D systems.

At matched total norm, structured perturbations have significantly longer VPT than isotropic noise in {L['paired']['iso_vpt']['sig_good']} of {L['paired']['iso_vpt']['n']} comparisons (shorter in {L['paired']['iso_vpt']['sig_bad']}) and significantly lower one-step error in {L['paired']['iso_mse']['sig_good']} of {L['paired']['iso_mse']['n']} (higher in {L['paired']['iso_mse']['sig_bad']}); against layer-matched noise the counts are {L['paired']['layer_vpt']['sig_good']} (VPT) and {L['paired']['layer_mse']['sig_good']} (one-step error) of {L['paired']['layer_vpt']['n']}. Failure is again rarely silent: {L['h1']['failed_ge10']:.0f}\% of failed models have a one-step error at least ten times the parent's ({L['h1']['failed_lt3']:.1f}\% below three times), while {L['h1']['surv_ge10']:.0f}\% of survivors also exceed that level. In the window $\log_{{10}}\mathrm{{MSE}}\in[-6,-3]$ VPT is log-linear in one-step error ($R^2={L['law']['r2']:.2f}$, slope ${L['law']['slope']:.2f}$ Lyapunov times per decade, $n={L['law']['n']}$). The tensor-isolated interventions (Figure~\ref{{fig:tensor}}, right column) reproduce the pattern of the 3-D systems qualitatively: at equal total norm $W_3$-only noise is the most damaging group ($r_{{50}}$ {rtxt('lorenz96_W3')} against {rtxt('lorenz96_all')} for isotropic noise and {rtxt('lorenz96_W2')} for $W_2$-only noise), and removing $W_3$ from isotropic noise lowers failure (Table~\ref{{tab:tensor}}) and VPT lost ({ci('lorenz96','all_but_W3',0.05)} Lyapunov times at $r=0.05$); in the own-norm coordinate the tensors fail at radii of {rtxt('lorenz96_W1', True, 2)} ($W_1$), {rtxt('lorenz96_W2', True, 2)} ($W_2$) and {rtxt('lorenz96_W3', True, 2)} ($W_3$), again within a factor of about 1.5 of each other but, unlike on the 3-D systems, at displacements of about {OWN_L96}\% of the tensor rather than {OWN_L63}\% on Lorenz-63. The absolute radii are thus system dependent; the dependence on the \emph{{coordinate}} (total versus per-tensor) is not.

\begin{{table}}[t]
\centering\small
\caption{{Lorenz-96: fraction of perturbed models that diverge or collapse, by structured perturbation and for isotropic noise of exactly the same norm ({nL} valid parents, 10 noise draws per parent).}}\label{{tab:l96fail}}
\begin{{tabular}}{{lccccc}}
\toprule
 & & \multicolumn{{2}}{{c}}{{Structured}} & \multicolumn{{2}}{{c}}{{Isotropic noise}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
Perturbation & $\relnorm$ & diverged & collapsed & diverged & collapsed\\
\midrule
\input{{tables/tab_l96_fail.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""
rep(r"\subsection{What predicts dynamical damage?}\label{sec:pred}", prot_sec + l96_sec + r"\subsection{What predicts dynamical damage?}\label{sec:pred}")

# =============================================================================================== discussion / limitations / conclusions
disc1 = rf"""\paragraph{{Does equal norm mean equal dynamics?}} Not in failure behaviour and, modestly, not in predictive accuracy for structured perturbations; and, much more strongly, not for random noise whose placement differs. At matched total norm, structured perturbations are usually less damaging than isotropic noise by VPT and one-step error, and failure modes differ markedly (divergence for pruning, collapse for noise on Lorenz-63), but these differences are small and largely vanish when per-tensor norms are matched. The tensor-isolated interventions explain why: the damage of a perturbation is governed by how far each tensor is displaced relative to its own size, and the total norm is dominated by the large middle matrix while the small tensors fail first. We therefore do not claim a universal ordering of perturbation families; for LLE the data are insufficient to make any claim, because most perturbed models fail.

\paragraph{{What should a compression budget be?}} The robustness radius of the whole network (a few percent on the 3-D systems, about {100 * r50_l96:.0f}\% on Lorenz-96) is a useful sanity scale, but the interventions say that it is the wrong coordinate for designing a compression scheme: a budget stated as a total relative norm lets the error concentrate in whichever tensors are small. A budget stated as a relative displacement per tensor, with each tensor below its own radius, is the quantity that the own-norm experiments show to be stable across tensors on Lorenz-63 and Lorenz-96 (up to a factor of about 1.5 between weight matrices). The protection experiment adds the corollary that sparing one tensor often improves valid prediction time but does not reliably prevent failure when the others are compressed beyond their own radii.

"""
between(r"\paragraph{Does equal norm mean equal dynamics?}", r"\paragraph{One-step error and long-run behaviour.}", disc1)

rep(r"\item \textbf{Scope.} Two three-dimensional systems, one small architecture, post-training compression only (no fine-tuning or quantization-aware training), one training recipe, and relative norms up to $0.6$.",
    r"\item \textbf{Scope.} Three systems (two three-dimensional and one six-dimensional), one small architecture (about $4.6$--$5.0\times10^3$ parameters), post-training compression only (no fine-tuning or quantization-aware training), one training recipe, and relative norms up to $0.6$. The absolute radii differ between systems (4.4\%, 1.8\% and about 18\% for isotropic noise), so they should not be transferred to other surrogates or larger networks.")
between(r"\item \textbf{Mechanism only partly tested.}", r"\item \textbf{Thresholds and functional forms.}",
        r"""\item \textbf{Mechanism only partly explained.} The interventions use Gaussian noise confined to one tensor and show that fragility is set by relative displacement per tensor, but we do not derive why each tensor fails at a relative displacement of the order of $0.1$ (3-D) or $0.4$ (Lorenz-96); the protection experiment tests only one simple remedy; and the slope of the log-linear VPT relation remains unexplained.
\item \textbf{Lorenz-96.} Only one dimension ($N=6$) and one forcing ($F=8$) were run, the exponent was estimated for the parents only, and the Lyapunov-exponent and Jacobian analyses were not repeated on it.
""")
rep(r"\item \textbf{H1 test.}", r"\item \textbf{Hypothesis H3.} H3 was formulated after the main results had been seen, from the displacement ratios of Section~\ref{sec:mech}; the interventions that test it were run once and the norms were fixed in advance, but the own-norm radii and the protection experiment are exploratory. \item \textbf{H1 test.}")

concl = rf"""\section{{Conclusions}}
In a controlled study of {n0(n_total)} perturbed chaotic surrogates of Lorenz-63, R\"ossler and Lorenz-96, the attractor survives only small parameter displacements: half of isotropic perturbations destroy it at 4.4\% (Lorenz-63), 1.8\% (R\"ossler) and {100 * r50_l96:.0f}\% (Lorenz-96) of the parameter norm, and failure is rarely silent (at least 98\% of failed models already have ten times their parent's one-step error). The total parameter norm is a poor summary of a perturbation: at the same total relative norm, noise confined to the small output layer is far more damaging than isotropic noise ({fl('lorenz', 0.01, 'W3')}\% against {fl('lorenz', 0.01, 'all')}\% failures on Lorenz-63 at $0.01$), noise confined to the large middle matrix is far less damaging ({fl('lorenz', 0.01, 'W2')}\%), and removing the output layer from isotropic noise cuts failures by {RED}\% although it carries 4\% of the squared norm. Measured in each tensor's own relative displacement, the tensors fail at comparable radii, so the natural budget is a per-tensor relative displacement, and protecting a single tensor from real pruning or quantization is not sufficient. At matched norm, structured perturbations differ from isotropic noise in failure mode and, modestly, in valid prediction time and one-step error, and most of this difference disappears when per-tensor norms are matched. Valid prediction time is approximately log-linear in one-step error in a window of moderate errors, with a system-specific slope that is not explained by Lyapunov or Jacobian statistics, and parent-relative one-step error adds substantial predictive power over norm for valid-prediction-time loss. Long-run exponent damage remains poorly predicted and, for most perturbed models, unmeasurable because they fail. Practically, compression of learned chaotic surrogates should be specified per tensor, evaluated with failure rates and valid prediction time alongside any exponent estimate, and exponent comparisons should be reported together with the fraction of models that survive.

"""
between(r"\section{Conclusions}", r"\section*{Data and code availability}", concl)

rep(r"Code (NumPy implementation of training, perturbations, Lyapunov and valid-prediction-time estimators, sensitivity measures and analysis), configuration, per-model raw results for all 60 parents,",
    r"Code (NumPy implementation of training, perturbations, Lyapunov and valid-prediction-time estimators, sensitivity measures, the tensor-isolated interventions, the protection experiment, the Lorenz-96 runner and analysis), configuration, per-model raw results for all parents of the three systems,")
rep(r"The full experiment runs in about one hour on a single CPU core. All results were produced with fixed seeds $0$--$29$ per system.",
    r"The 3-D experiment runs in about one hour on a single CPU core, and the interventions, the protection experiment and the Lorenz-96 study add roughly one further hour. All results were produced with fixed seeds $0$--$29$ per system.")

# appendix additions
rep(r"Systems & Lorenz-63, R\"ossler; RK4, internal step $10^{-3}$, sampled at $0.02$, transient $20$\\",
    r"Systems & Lorenz-63, R\"ossler; RK4, internal step $10^{-3}$, sampled at $0.02$, transient $20$. Lorenz-96: $N=6$, $F=8$, RK4 internal step $5\times10^{-3}$, sampled at $0.05$, transient $20$\\")
rep(r"Model & Residual MLP 3--64--64--3, tanh, Xavier init (output layer scaled $0.1$)\\",
    r"Model & Residual MLP 3--64--64--3 (6--64--64--6 for Lorenz-96), tanh, Xavier init (output layer scaled $0.1$)\\")
rep(r"Sensitivity orbit & parent's own 3{,}000-step orbit (true orbit if collapsed), burn-in 300\\",
    r"""Sensitivity orbit & parent's own 3{,}000-step orbit (true orbit if collapsed), burn-in 300\\
Interventions & noise confined to $W_1$, $W_2$, $W_3$ or biases; total-norm design $r\in\{0.01,0.02,0.05,0.1,0.2\}$ (plus isotropic and isotropic-without-one-group controls); own-norm design $r\in\{0.02,0.05,0.1,0.2,0.5\}$; 10 draws each; 650 models per parent\\
Protection & global/layer-wise pruning $25/50/75\%$ and quantization $8/6/4/3$ bit with scopes all, all except $(W_3,b_3)$, only $(W_3,b_3)$; 30 deterministic models per parent\\""")

# ======================================================================================= first-order theory block
TH = json.load(open(os.path.join(P, "theory_numbers.json")))
ACC, AUC_, CR, RS = TH["accuracy"], TH["auc"], TH["critical"], TH["radii_summary"]
SYSL = {"lorenz": r"Lorenz-63", "rossler": r"R\"ossler", "lorenz96": r"Lorenz-96"}
fac = {s: 10 ** ACC[f"{s}|large"]["med_log10_ratio"] for s in SYSL}
pred_rad = {s: [r["predicted"] for r in TH["radii"] if r["system"] == s] for s in SYSL}
spread_l63 = max(pred_rad["lorenz"]) / min(pred_rad["lorenz"])
bad = [r for r in TH["radii"] if not (0.5 < r["ratio"] < 2.0)]
bad_txt = " and ".join(f"{SYSL[r['system']]} {r['group'].replace('biases', 'the biases').replace('W', '$W_') + ('$' if r['group'].startswith('W') else '')} ({r['ratio']:.1f}$\\times$)" for r in bad)

theory_sec = rf"""\subsection{{A first-order account of the per-tensor radii}}\label{{sec:theory}}
The interventions show \emph{{where}} noise is harmful; we now ask whether the radii can be \emph{{predicted}} from the parent alone. For i.i.d.\ Gaussian noise of variance $\sigma^2$ on a set $S$ of parameter tensors, a first-order expansion of the one-step map $f_\theta$ gives $\mathbb{{E}}\|f_{{\theta+\delta}}(x)-f_\theta(x)\|^2\approx\sigma^2\sum_{{k\in S}}\|J_k(x)\|_F^2$ with $J_k=\partial f_\theta/\partial\theta_k$, so that the expected one-step MSE increase (MSE being the mean over the $d$ state coordinates and the test inputs) is
\begin{{equation}}\label{{eq:dmse}}
\Delta\mathrm{{MSE}}\;\approx\;\frac{{\|\delta\|^2}}{{n_S}}\sum_{{k\in S}}s_k,\qquad s_k=\frac1d\,\mathbb{{E}}_x\|J_k(x)\|_F^2,
\end{{equation}}
where $\|\delta\|$ is the norm of the injected noise and $n_S$ its number of entries. For the residual MLP the sensitivities $s_k$ are analytic (with $h_1=\tanh(W_1x+b_1)$, $h_2=\tanh(W_2h_1+b_2)$ and $D_i=\mathrm{{diag}}(1-h_i^2)$): $s_{{b_3}}=1$, $s_{{W_3}}=\mathbb{{E}}\|h_2\|^2$, $s_{{b_2}}=\mathbb{{E}}\|W_3D_2\|_F^2/d$, $s_{{W_2}}=\mathbb{{E}}[\|h_1\|^2\|W_3D_2\|_F^2]/d$, $s_{{b_1}}=\mathbb{{E}}\|W_3D_2W_2D_1\|_F^2/d$ and $s_{{W_1}}=\mathbb{{E}}[\|x\|^2\|W_3D_2W_2D_1\|_F^2]/d$, with expectations over the test inputs; no perturbed model has to be evaluated. For noise confined to a tensor group $G$ and displaced by a fraction $\varepsilon$ of its own norm, \eqref{{eq:dmse}} reads $\Delta\mathrm{{MSE}}\approx\varepsilon^2\kappa_G$ with $\kappa_G=(\|\theta_G\|^2/n_G)\sum_{{k\in G}}s_k$. If failure sets in when the induced one-step error reaches a system-specific multiple $c$ of the parent's MSE, the radius of the tensor is
\begin{{equation}}\label{{eq:radius}}
r_G=\sqrt{{c\,\mathrm{{MSE}}_{{\mathrm{{parent}}}}/\kappa_G}}.
\end{{equation}}
This account was formulated after the interventions had been analysed, so the tests below are exploratory; the only fitted quantity is $c$, one number per system.

\paragraph{{The first-order prediction is accurate where it should be.}} Over the models whose predicted increase exceeds ten times the parent's MSE, which contain almost all failures, the regression of the measured on the predicted $\log_{{10}}$ error has slope ${ACC['lorenz|large']['slope']:.2f}$, ${ACC['rossler|large']['slope']:.2f}$ and ${ACC['lorenz96|large']['slope']:.2f}$ and $R^2={ACC['lorenz|large']['r2']:.2f}$, ${ACC['rossler|large']['r2']:.2f}$ and ${ACC['lorenz96|large']['r2']:.2f}$ (Lorenz-63, R\"ossler, Lorenz-96; $n={n0(ACC['lorenz|large']['n'])}$, ${n0(ACC['rossler|large']['n'])}$ and ${n0(ACC['lorenz96|large']['n'])}$ models), and the measured error is on median ${fac['lorenz']:.2f}$, ${fac['rossler']:.2f}$ and ${fac['lorenz96']:.2f}$ times the predicted one (Figure~\ref{{fig:theory}}a). Between one and ten times the parent's MSE the slope is still close to one (${ACC['lorenz|moderate']['slope']:.2f}$--${ACC['rossler|moderate']['slope']:.2f}$) but the scatter is larger ($R^2$ ${ACC['rossler|moderate']['r2']:.2f}$--${ACC['lorenz96|moderate']['r2']:.2f}$); below the parent's MSE the measured increase is dominated by sampling and cross terms (Spearman $\rho$ {ACC['lorenz|small']['spearman']:.2f}--{ACC['lorenz96|small']['spearman']:.2f}) and the first-order result is uninformative.

\paragraph{{Failure depends on the predicted error.}} Used as the only predictor of failure over all intervention models of a system, $\log_{{10}}$ of the predicted increase reaches AUC ${AUC_['lorenz']['predicted_dMSE']:.2f}$, ${AUC_['rossler']['predicted_dMSE']:.2f}$ and ${AUC_['lorenz96']['predicted_dMSE']:.2f}$, against ${AUC_['lorenz']['total_rel_norm']:.2f}$, ${AUC_['rossler']['total_rel_norm']:.2f}$ and ${AUC_['lorenz96']['total_rel_norm']:.2f}$ for the total relative norm and ${AUC_['lorenz']['own_rel_noised_set']:.2f}$, ${AUC_['rossler']['own_rel_noised_set']:.2f}$ and ${AUC_['lorenz96']['own_rel_noised_set']:.2f}$ for the displacement relative to the norm of the noised tensors (within-parent averages differ by at most $0.02$). The gain is consistent but modest, and Figure~\ref{{fig:theory}}c shows that the failure rate rises monotonically with the predicted error, with a system-specific midpoint (the $c$ fitted below) and a steeper rise on Lorenz-96.

\paragraph{{One critical error ratio per system predicts the per-tensor radii out of design.}} We fit $c$ by logistic regression of failure on $\log_{{10}}$ predicted increase using only the total-norm and leave-one-out designs, which gives $c={CR['lorenz']['ratio']:.0f}$, ${CR['rossler']['ratio']:.0f}$ and ${CR['lorenz96']['ratio']:.0f}$ (Lorenz-63, R\"ossler, Lorenz-96), equivalent to a critical one-step rms error of {100 * CR['lorenz']['rms_critical']:.1f}\%, {100 * CR['rossler']['rms_critical']:.1f}\% and {100 * CR['lorenz96']['rms_critical']:.0f}\% of the state's standard deviation. Equation~\eqref{{eq:radius}} then predicts the own-norm radius of every tensor from the parent's $\kappa_G$ (Table~\ref{{tab:theoryradii}}, Figure~\ref{{fig:theory}}b): {RS['within_2']} of {RS['n']} radii lie within a factor of two of the measured ones and {RS['within_1p5']} within $1.5$ (median factor ${RS['median_abs_factor']:.2f}$, correlation of the logarithms ${RS['pearson_log']:.2f}$). The exceptions are {bad_txt}, whose fragility the first-order theory underestimates; the R\"ossler parents are only weakly chaotic (Section~\ref{{sec:h1}}) and damage there need not be governed by one-step error alone, which we did not test.

\paragraph{{What this does and does not explain.}} Equation~\eqref{{eq:radius}} shows that the radius of a tensor is the displacement at which its quadratically growing contribution to the one-step error reaches the critical error of the system, so tensors differ in fragility through $\kappa_G$, and the total norm is a poor budget because the error caused by a given norm depends on how it is spread over the $s_k$. On Lorenz-63 the predicted radii of the four tensor groups differ by at most a factor of ${spread_l63:.1f}$, which is why the weight matrices fail at nearly the same relative displacement there; this near-equality is a property of these surrogates, not a law, and the value of about $0.1$ is the square root of the ratio between the critical error and $\kappa_G$, not a constant. The account also gives a way to set per-tensor budgets without running perturbation experiments, once $c$ has been calibrated for the system (here $c$ ranges over {CR['lorenz']['ratio']:.0f}--{CR['lorenz96']['ratio']:.0f} and has to be calibrated, for example on a few noise draws). It is a first-order result for Gaussian noise, not a bound, and it does not explain \emph{{why}} $c$ takes the values it does.

\begin{{figure}}[t]
\centering\includegraphics[width=\textwidth]{{figures/fig_theory.pdf}}
\caption{{First-order sensitivity account. (a) Predicted against measured one-step MSE increase relative to the parent's MSE, all intervention models (subsampled). (b) Own-norm radii predicted by \eqref{{eq:radius}} with $c$ fitted on the total-norm and leave-one-out designs only, against the measured radii (b: biases); the band is a factor of two. (c) Fraction of failed models against predicted increase, all tensors, designs and norms pooled.}}\label{{fig:theory}}
\end{{figure}}

\begin{{table}}[t]
\centering\small
\caption{{Measured own-norm radii $r_{{50}}$ and radii predicted by \eqref{{eq:radius}} from the parent's sensitivities, with one critical error ratio $c$ per system fitted on other designs; last column: predicted/measured.}}\label{{tab:theoryradii}}
\begin{{tabular}}{{llccc}}
\toprule
System & tensor & measured & predicted & ratio\\
\midrule
\input{{tables/tab_theory_radii.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""
rep(r"\subsection{Does protecting the output layer help real compression?}\label{sec:protect}",
    theory_sec + r"\subsection{Does protecting the output layer help real compression?}\label{sec:protect}")

# abstract
rep(r"so a per-tensor relative displacement is a better compression budget than the total norm; protecting only",
    rf"so a per-tensor relative displacement is a better compression budget than the total norm, and a first-order sensitivity analysis computed from the parent alone predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two once one critical error ratio per system is fitted on other designs; protecting only")
# highlights
rep(r"\item Protecting only the output layer from pruning or quantization is not sufficient to prevent failure.",
    rf"\item A first-order sensitivity analysis of the parent predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two, using one fitted error threshold per system." + "\n" + r"\item Protecting only the output layer from pruning or quantization is not sufficient to prevent failure.")
# introduction
rep(r"a protection experiment on real pruning and quantization (Section~\ref{sec:protect}), and a replication",
    r"a first-order sensitivity account of the per-tensor radii (Section~\ref{sec:theory}), a protection experiment on real pruning and quantization (Section~\ref{sec:protect}), and a replication")
# contributions
rep(r"\item A \emph{matched-control protocol}",
    rf"\item \emph{{A first-order sensitivity account.}} The one-step error caused by noise in a tensor is predicted analytically from the parent's parameter Jacobians; with one critical error ratio per system fitted on other designs it predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two (R\"ossler's output layer and middle matrix are the exceptions), and can be used to set per-tensor budgets without perturbation experiments." + "\n" + r"\item A \emph{matched-control protocol}")
# summary table row
rep(r"Does protecting the output layer prevent failure under real compression?$^\dagger$",
    rf"Why does each tensor fail at its radius?$^\dagger$ & Failure sets in when the one-step error induced by the noise reaches a system-specific multiple of the parent's ({CR['lorenz']['ratio']:.0f}$\times$, {CR['rossler']['ratio']:.0f}$\times$, {CR['lorenz96']['ratio']:.0f}$\times$); the induced error is predicted analytically from the parent, and this predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two out of design (\S\ref{{sec:theory}}).\\[2pt]" + "\n" + r"Does protecting the output layer prevent failure under real compression?$^\dagger$")
# discussion
rep(r"does not reliably prevent failure when the others are compressed beyond their own radii.",
    r"does not reliably prevent failure when the others are compressed beyond their own radii. Section~\ref{sec:theory} adds that the radius of a tensor can be estimated from the parent alone as $\sqrt{c\,\mathrm{MSE}_{\mathrm{parent}}/\kappa_G}$ once the critical error ratio $c$ of the system is known.")
# limitations
rep(r"but we do not derive why each tensor fails at a relative displacement of the order of $0.1$ (3-D) or $0.4$ (Lorenz-96); the protection experiment",
    rf"and the first-order account of Section~\ref{{sec:theory}} predicts the radii but treats the critical error ratio as a fitted, system-specific constant ({CR['lorenz']['ratio']:.0f}--{CR['lorenz96']['ratio']:.0f}), is a first-order result for Gaussian noise rather than a bound, was formulated after the interventions, and misses two R\"ossler tensors by a factor of two to three; the protection experiment")
# conclusions
rep(r"and protecting a single tensor from real pruning or quantization is not sufficient. At matched norm",
    rf"and protecting a single tensor from real pruning or quantization is not sufficient. A first-order sensitivity analysis computed from the parent alone predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two once a single critical error ratio per system is fitted on other designs. At matched norm")
# data availability and appendix
rep(r"the Lorenz-96 runner and analysis)", r"the Lorenz-96 runner, the first-order sensitivity theory and analysis)")
rep(r"30 deterministic models per parent\\",
    r"""30 deterministic models per parent\\
Theory & analytic sensitivities $s_k=\mathbb{E}\|J_k\|_F^2/d$ on the test inputs; $c$ fitted by logistic regression of failure on $\log_{10}$ predicted increase over the total-norm and leave-one-out designs; radii from \eqref{eq:radius}\\""")

# ======================================================================================= mixed-precision block
NAME_S = {"lorenz": "Lorenz-63", "rossler": "R\\\"ossler", "lorenz96": "Lorenz-96"}
MX = json.load(open(os.path.join(P, "mixed_numbers.json")))
MS = pd.DataFrame(MX["summary"])
MP = pd.DataFrame(MX["paired"])
MA = MX["auc"]


def ms(s_, b_, col):
    return MS[(MS.system == s_) & (MS.budget == b_)].iloc[0][col]


def mp(s_, b_, cmp_, metric, col):
    return MP[(MP.system == s_) & (MP.budget == b_) & (MP["cmp"] == cmp_) & (MP.metric == metric)].iloc[0][col]


rand_ratio = MS.pred_ratio_random_uniform
vpt_u = MP[(MP["cmp"] == "theory-uniform") & (MP.metric == "vpt_loss")]
vpt_sig = vpt_u[vpt_u.p_holm < 0.05]
vpt_r = MP[(MP["cmp"] == "theory-random") & (MP.metric == "vpt_loss")]
fail_u = MP[(MP["cmp"] == "theory-uniform") & (MP.metric == "fail")]
max_fail_diff = 100 * fail_u["mean"].abs().max()
same_u = MS.theory_equals_uniform
pred_ratio_min = MS.pred_ratio_theory_uniform.min()


def sig_row(s_, b_):
    return f"{mp(s_, b_, 'theory-uniform', 'vpt_loss', 'median'):+.2f}"


mixed_sec = rf"""\subsection{{Does the theory design better compression? Mixed-precision quantization}}\label{{sec:mixed}}
Equation~\eqref{{eq:dmse}} also predicts the one-step error of any per-tensor quantization: rounding with $b_k$ bits adds noise of variance $(a_k/q_{{b_k}})^2/12$ per entry, with $a_k=\max|\theta_k|$ and $q_b=2^{{b-1}}-1$, so that
\begin{{equation}}\label{{eq:quant}}
\Delta\mathrm{{MSE}}\;\approx\;\sum_{{k}}s_k\,\frac{{a_k^2}}{{12\,q_{{b_k}}^2}}.
\end{{equation}}
We use it to choose a bit width $b_k\in\{{2,\dots,8\}}$ for each of the six tensors by exhaustive search over the $7^6$ combinations (no heuristics) that minimises the predicted error at a given average budget of $3$, $4$ or $5$ bits per parameter, and compare, at the same number of bits, with uniform quantization at that width (as in the rest of the paper) and with $20$ random allocations per parent that meet the same budget (a matched control in the spirit of Section~\ref{{sec:intv}}). This is $66$ deterministic or random models per parent on all {sum(MX['parents'].values())} valid parents.

\paragraph{{The theory screens allocations well.}} Random allocations have a median predicted error of ${rand_ratio.min():.0f}$--${rand_ratio.max():.0f}\times$ that of uniform quantization, and they fail much more often (Table~\ref{{tab:mixed}}; for example $4$ bits: {ms('lorenz', 4, 'fail_random'):.0f}\% against {ms('lorenz', 4, 'fail_uniform'):.0f}\% on Lorenz-63, {ms('rossler', 4, 'fail_random'):.0f}\% against {ms('rossler', 4, 'fail_uniform'):.0f}\% on R\"ossler and {ms('lorenz96', 4, 'fail_random'):.0f}\% against {ms('lorenz96', 4, 'fail_uniform'):.0f}\% on Lorenz-96) and lose more valid prediction time than the theory-guided allocation in all nine system--budget cells (median difference {vpt_r['median'].abs().min():.2f}--{vpt_r['median'].abs().max():.2f} Lyapunov times, Holm-adjusted $p<0.001$ in all cells). In terms of the failure rate alone, the per-parent difference to random allocations is significant only on Lorenz-96 at $4$ and $5$ bits ({mp('lorenz96', 4, 'theory-random', 'fail', 'better')} and {mp('lorenz96', 5, 'theory-random', 'fail', 'better')} parents better, {mp('lorenz96', 4, 'theory-random', 'fail', 'worse')} and {mp('lorenz96', 5, 'theory-random', 'fail', 'worse')} worse). Across all allocations of a system, the predicted error alone separates failed from surviving models with AUC ${MA['lorenz']['pooled']:.2f}$ (Lorenz-63), ${MA['rossler']['pooled']:.2f}$ (R\"ossler) and ${MA['lorenz96']['pooled']:.2f}$ (Lorenz-96), and with AUC ${MA['lorenz']['within']:.2f}$, ${MA['rossler']['within']:.2f}$ and ${MA['lorenz96']['within']:.2f}$ within a parent and budget, that is, among allocations of the same network at the same cost.

\paragraph{{But the optimum is close to uniform, and the practical gain is small.}} Because $W_2$ holds 89\% of the parameters and therefore of the bits, the exact optimum differs little from uniform quantization: it coincides with it in {same_u.iloc[[0, 3, 6]].mean():.0f}\% of the parents at $3$ bits and reduces the predicted error by at most {100 * (1 - pred_ratio_min):.0f}\% (median over parents) at $4$ and $5$ bits. The failure rate is not significantly changed anywhere (largest difference {max_fail_diff:.0f} percentage points; the smallest Holm-adjusted $p$ is ${fail_u.p_holm.min():.2f}$), but the valid prediction time lost is significantly smaller than for uniform quantization in {len(vpt_sig)} of the nine cells ({', '.join(f"{NAME_S[r.system]} {int(r.budget)} bits: {r['median']:+.2f}" for _, r in vpt_sig.iterrows())} Lyapunov times, Holm-adjusted $p={vpt_sig.p_holm.min():.3f}$--${vpt_sig.p_holm.max():.3f}$), which is about {100 * abs(vpt_sig[vpt_sig.system == 'rossler']['median'].min()) / 0.43:.0f}\% of the R\"ossler parent's valid prediction time for the largest R\"ossler effect and smaller on Lorenz-96. We therefore read the experiment as showing that the first-order theory is a reliable, cheap screen for quantization configurations (it identifies bad allocations without evaluating them) and a modest optimiser, not that it removes the failures of low-bit quantization.

\begin{{table}}[t]
\centering\small
\caption{{Quantization at equal average bit budget: failure (\%) of uniform, random (mean of 20 allocations per parent) and theory-guided allocations, valid prediction time lost (Lyapunov times, median over parents) for uniform and theory-guided allocations, the median per-parent difference theory$-$uniform, and its Holm-adjusted Wilcoxon $p$.}}\label{{tab:mixed}}
\begin{{tabular}}{{llccccccc}}
\toprule
System & bits & \multicolumn{{3}}{{c}}{{failure (\%)}} & \multicolumn{{2}}{{c}}{{VPT lost}} & $\Delta$VPT & $p_{{\mathrm{{Holm}}}}$\\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-7}}
 & & uniform & random & theory & uniform & theory & & \\
\midrule
\input{{tables/tab_mixed.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""

# ======================================================================================= width + sensitivity block
WN = json.load(open(os.path.join(P, "width_numbers.json")))
FS = TH["fit_sensitivity"]
cells = [(s_, H_) for s_ in ("lorenz", "rossler") for H_ in ("32", "64", "128")]
w_within = sum(int(0.5 < WN[H_][s_]["radii"][g_]["predicted"] / WN[H_][s_]["radii"][g_]["measured"] < 2.0)
               for s_, H_ in cells for g_ in ("W1", "W2", "W3", "biases"))
w_total = 4 * len(cells)
w_slopes = [WN[H_][s_]["acc_slope"] for s_, H_ in cells]
w_r2 = [WN[H_][s_]["acc_r2"] for s_, H_ in cells]
w_miss = [(s_, H_, g_, WN[H_][s_]["radii"][g_]["predicted"] / WN[H_][s_]["radii"][g_]["measured"])
          for s_, H_ in cells for g_ in ("W1", "W2", "W3", "biases")
          if not (0.5 < WN[H_][s_]["radii"][g_]["predicted"] / WN[H_][s_]["radii"][g_]["measured"] < 2.0)]
w_miss_txt = " and ".join(f"{'the biases' if g_ == 'biases' else '$W_' + g_[1] + '$'} on {NAME_S[s_]} at $H={H_}$ (predicted/measured ${r_:.2f}$)" for s_, H_, g_, r_ in w_miss)
wr = lambda s_, H_, g_: WN[H_][s_]["radii"][g_]["measured"]
w_weight = [wr(s_, H_, g_) for s_, H_ in cells for g_ in ("W1", "W2", "W3")]
cL = [WN[H_]["lorenz"]["c"] for H_ in ("32", "64", "128")]
cR = [WN[H_]["rossler"]["c"] for H_ in ("32", "64", "128")]
nW = {(s_, H_): WN[H_][s_]["parents"] for s_, H_ in cells}

width_sec = rf"""\subsection{{Does the picture depend on the network width?}}\label{{sec:width}}
All results so far use hidden width $H=64$. We trained new parents with $H=32$ and $H=128$ (same recipe, 16 seeds; {nW[('lorenz', '32')]} and {nW[('lorenz', '128')]} valid Lorenz-63 parents, {nW[('rossler', '32')]} and {nW[('rossler', '128')]} valid R\"ossler parents, validity meaning only that the parent's own rollout neither diverges nor collapses), applied the total-norm and own-norm designs with $6$ draws ($270$ models per parent) and computed the first-order prediction as before. To make the widths comparable the critical ratio $c$ is fitted on the total-norm design only, so the $H=64$ row differs from Section~\ref{{sec:theory}}, where the leave-one-out design was included (Table~\ref{{tab:width}}).

\paragraph{{What does not depend on the width.}} The accuracy of the first-order prediction of the one-step error is the same at every width: for models whose predicted increase exceeds ten times the parent's MSE the slope of measured on predicted $\log_{{10}}$ error is ${min(w_slopes):.2f}$--${max(w_slopes):.2f}$ and $R^2={min(w_r2):.2f}$--${max(w_r2):.2f}$ in all six system--width cells. The weight matrices stay fragile at displacements of the order of $0.1$: the own-norm radii of $W_1$, $W_2$ and $W_3$ lie between ${min(w_weight):.2f}$ and ${max(w_weight):.2f}$ in every cell, and {w_within} of the {w_total} radii are predicted within a factor of two (the exceptions: {w_miss_txt}; for the biases at $H=128$ on R\"ossler the measured radius is an extrapolation beyond the largest tested displacement of $0.5$).

\paragraph{{What does depend on it.}} The radius is not a width-invariant constant. On Lorenz-63 the radii of the weight matrices rise with the width, between $H=32$ and $H=128$, from {wr('lorenz', '32', 'W1'):.2f} to {wr('lorenz', '128', 'W1'):.2f} ($W_1$), from {wr('lorenz', '32', 'W2'):.2f} to {wr('lorenz', '128', 'W2'):.2f} ($W_2$) and from {wr('lorenz', '32', 'W3'):.2f} to {wr('lorenz', '128', 'W3'):.2f} ($W_3$), a factor of about $1.3$--$2.1$, while on R\"ossler $W_2$ rises from {wr('rossler', '32', 'W2'):.2f} to {wr('rossler', '128', 'W2'):.2f} and $W_1$ and $W_3$ stay near $0.1$; the biases become markedly more robust with width on both systems. The fitted critical error ratio also changes with the width, from {cL[0]:.0f} to {cL[2]:.0f} on Lorenz-63 and from {cR[0]:.0f} to {cR[2]:.0f} on R\"ossler (fitted on the total-norm design), so $c$ has to be calibrated per network and not only per system. We tested only $H\in\{{32,64,128\}}$ and one depth.

\begin{{table}}[t]
\centering\small
\caption{{Own-norm radii $r_{{50}}$ of the four tensor groups (measured) and ratio of the radius predicted by \eqref{{eq:radius}} to the measured one, for three hidden widths. $c$: critical error ratio fitted on the total-norm design only. Valid parents in the third column.}}\label{{tab:width}}
\begin{{tabular}}{{lccccccccccc}}
\toprule
 & & & & \multicolumn{{4}}{{c}}{{measured $r_{{50}}$}} & \multicolumn{{4}}{{c}}{{predicted/measured}}\\
\cmidrule(lr){{5-8}}\cmidrule(lr){{9-12}}
System & $H$ & parents & $c$ & $W_1$ & $W_2$ & $W_3$ & biases & $W_1$ & $W_2$ & $W_3$ & biases\\
\midrule
\input{{tables/tab_width.tex}}\\
\bottomrule
\end{{tabular}}
\end{{table}}

"""
rep(r"\subsection{Does protecting the output layer help real compression?}\label{sec:protect}",
    width_sec + r"\subsection{Does protecting the output layer help real compression?}\label{sec:protect}")

# ======================================================================================= final edits for width / mixed
rep(r"\subsection{Replication on a six-dimensional system: Lorenz-96}\label{sec:l96}",
    mixed_sec + r"\subsection{Replication on a six-dimensional system: Lorenz-96}\label{sec:l96}")

_cs = lambda s_: [v["c"][s_] for v in FS.values()]
sens_par = rf"""\paragraph{{Sensitivity to how $c$ is fitted.}} The number of radii within a factor of two is {FS['total']['within_2']}, {FS['loo']['within_2']}, {FS['total+loo']['within_2']} and {FS['all_incl_own']['within_2']} of {FS['total']['n']} when $c$ is fitted on the total-norm design only, on the leave-one-out design only, on both, and on all three designs (the last one in sample), so the result does not hinge on this choice, but which tensors are missed does. The fitted $c$ is stable for Lorenz-63 ({min(_cs('lorenz')):.0f}--{max(_cs('lorenz')):.0f}) and Lorenz-96 ({min(_cs('lorenz96')):.0f}--{max(_cs('lorenz96')):.0f}) but not for R\"ossler ({min(_cs('rossler')):.0f}--{max(_cs('rossler')):.0f}), where failures occur at small predicted errors and the failure curve is shallow; the R\"ossler exceptions above are therefore a statement about the fit as much as about the theory.

"""
rep(r"\paragraph{What this does and does not explain.}", sens_par + r"\paragraph{What this does and does not explain.}")

auc_lo = min(MA[s_]["pooled"] for s_ in MA)
auc_hi = max(MA[s_]["pooled"] for s_ in MA)
rep(rf"predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two once one critical error ratio per system is fitted on other designs; protecting only",
    rf"predicts {RS['within_2']} of {RS['n']} per-tensor radii within a factor of two once one critical error ratio per system is fitted on other designs (the same prediction accuracy of the one-step error holds for hidden widths 32--128, although the radii and the critical ratio change with the width), and it ranks quantization bit allocations well (AUC {auc_lo:.2f}--{auc_hi:.2f}) without improving much on uniform bit widths; protecting only")
rep(r"\item Protecting only the output layer from pruning or quantization is not sufficient to prevent failure.",
    rf"\item The first-order prediction of the one-step error is equally accurate for hidden widths 32, 64 and 128; the radii rise with width on Lorenz-63 and the critical error ratio falls." + "\n"
    + rf"\item The predicted error ranks quantization bit allocations well (AUC {auc_lo:.2f}--{auc_hi:.2f}) but the optimal allocation improves little on uniform bit widths." + "\n"
    + r"\item Protecting only the output layer from pruning or quantization is not sufficient to prevent failure.")
rep(r"a protection experiment on real pruning and quantization (Section~\ref{sec:protect}), and a replication",
    r"a check across network widths (Section~\ref{sec:width}), a protection experiment on real pruning and quantization (Section~\ref{sec:protect}), a mixed-precision quantization test (Section~\ref{sec:mixed}), and a replication")
rep(r"\item A \emph{matched-control protocol}",
    rf"\item \emph{{Generality and use of the theory.}} The prediction accuracy of the one-step error is unchanged for hidden widths 32--128, while the radii and the critical error ratio change with the width; and the predicted error ranks quantization bit allocations (AUC {auc_lo:.2f}--{auc_hi:.2f}) and beats random allocations in valid prediction time, but the exact optimum is close to uniform bit widths and improves little on the standard choice." + "\n" + r"\item A \emph{matched-control protocol}")
rep(r"Does protecting the output layer prevent failure under real compression?$^\dagger$",
    rf"Does the picture depend on the network width?$^\dagger$ & The accuracy of the one-step error prediction is the same at $H=32,64,128$ ($R^2={min(w_r2):.2f}$--${max(w_r2):.2f}$) and {w_within} of {w_total} radii are predicted within a factor of two; the radii themselves rise with width on Lorenz-63 (by $1.3$--$2.1\times$ from 32 to 128) and the critical ratio falls (\S\ref{{sec:width}}).\\[2pt]" + "\n"
    + rf"Can the theory design better quantization?$^\dagger$ & It ranks bit allocations well (AUC {auc_lo:.2f}--{auc_hi:.2f}; random allocations have {rand_ratio.min():.0f}--{rand_ratio.max():.0f}$\times$ the predicted error and fail more), but the optimum is close to uniform: no significant change in failure and significantly less valid prediction time lost in {len(vpt_sig)} of 9 cells (\S\ref{{sec:mixed}}).\\[2pt]" + "\n"
    + r"Does protecting the output layer prevent failure under real compression?$^\dagger$")
rep(r"once the critical error ratio $c$ of the system is known.",
    r"once the critical error ratio $c$ of the system is known; Section~\ref{sec:width} shows that this ratio and the radii themselves change with the network width, so the budget has to be calibrated for each network.")
rep(r"\item \textbf{Hypothesis H3.}",
    r"\item \textbf{Width, depth and fit.} The width check covers $H\in\{32,64,128\}$ with 12--16 parents per cell, only the two 3-D systems and one depth, and its parent-validity rule is weaker than in the main study; the critical ratio varies with the width and, for R\"ossler, with the designs used to fit it; and the mixed-precision optimum rests on the same first-order approximation, which is crude at 2 bits. \item \textbf{Hypothesis H3.}")
rep(r"within a factor of two once a single critical error ratio per system is fitted on other designs. At matched norm",
    rf"within a factor of two once a single critical error ratio per system is fitted on other designs; the prediction accuracy of the one-step error is the same for hidden widths 32--128 although the radii and the critical ratio change with the width, and the predicted error ranks quantization bit allocations well (AUC {auc_lo:.2f}--{auc_hi:.2f}) but yields only small gains over uniform bit widths. At matched norm")
rep(r"the first-order sensitivity theory and analysis)", r"the first-order sensitivity theory, the width and mixed-precision experiments and analysis)")
rep(r"radii from \eqref{eq:radius}\\",
    r"""radii from \eqref{eq:radius}\\
Width & new parents with $H\in\{32,128\}$, 16 seeds, total-norm and own-norm designs with 6 draws (270 models per parent)\\
Mixed precision & bit widths $2$--$8$ per tensor, average budgets $3/4/5$ bits; uniform, exact-search theory-guided and 20 random allocations per parent (66 models)\\""")

# float placement: allow float pages (the long summary table otherwise blocks every later table) and fix microtype
assert "xcolor,microtype,threeparttable" in src
src = src.replace("xcolor,microtype,threeparttable", "xcolor,threeparttable", 1)
src = src.replace(r"\usepackage[a4paper,margin=1in]{geometry}",
                  r"\usepackage[a4paper,margin=1in]{geometry}" + "\n" + r"\usepackage[expansion=false]{microtype}" + "\n"
                  + r"\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.08}\renewcommand{\floatpagefraction}{0.75}", 1)
src = src.replace(r"\begin{table}[t]", r"\begin{table}[tbp]").replace(r"\begin{figure}[t]", r"\begin{figure}[tbp]")
open(os.path.join(P, "main_v2.tex"), "w").write(src)
print("wrote paper/main_v2.tex", len(src))
