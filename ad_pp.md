---
title: "A tale of two cities"
author: McCoy R. Becker
date: Friday, April 24, 2020
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  - \usepackage{listings}
  - \usepackage{libertine}
  - \usepackage{soul}
  - \usepackage{tabularx}
  - \lstset{language=C}
  - \hypersetup{colorlinks=true}
  - \setbeamercolor{normal text}{fg=white,bg=black!85}
  - \setbeamercolor*{structure}{fg=blue!20!white}
  - \setbeamercolor{alerted text}{use=structure,fg=structure.fg}
  - \setbeamercolor*{palette primary}{use=structure,fg=structure.fg}
  - \setbeamercolor*{palette secondary}{use=structure,fg=structure.fg!95!black}
  - \setbeamercolor*{palette tertiary}{use=structure,fg=structure.fg!90!black}
  - \setbeamercolor*{palette quaternary}{use=structure,fg=structure.fg!95!black,bg=black!80}
  - \setbeamercolor*{framesubtitle}{fg=white}
bibliography: bibliography.bib

---

#

> It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.

Charles Dickens, 1859

#

This is a modified version of a discussion I gave last week to the compiler group. I've slimmed this version down, and made it slightly more opinionated.

\vspace{0.5cm}

> The expression of automatic differentiation and probabilistic programming systems rely on a similar set of programming patterns. These can be identified by thinking about the 'runtimes' of these systems and what representations they work with. In particular, you'll find much in common between these systems and compilers/interpreters.

This is survey oriented.

# Modern AI

\begin{itemize}
\item Automatic differentiation
\item Probabilistic programming
\item Discrete optimization
\end{itemize}

This is essentially the whole story. So why aren't we _there_ yet?

\hrulefill

*_there_ is artificial general intelligence.

# Stuck at local optima (in abstraction space)?


\begin{itemize}
\item Deep learning: core abstraction is the computation graph, which represents the flow of differentiable information in your program. You fill the graph up with tensors. Deep learning engineers are paid large amounts to build these graphs.
\end{itemize}

\begin{figure}
\includegraphics[width=5cm]{machine_learning_2x.png}
\end{figure}

# Stuck at local optima (in abstraction space)?

\begin{itemize}
\item Probabilistic programming: core abstraction is either a form of dataflow graph (i.e. an explicit network representation of a model) or a dynamic structure representing a program trace, depending on the probabilistic programming system.
\end{itemize}

\begin{figure}
\centering
\includegraphics[width=11cm]{mdp.jpeg}
\end{figure}

\hrulefill

In the flow graph case, the information content of the graph is different compared to deep learning - the connections represent the conditional dependencies.

# Long tale short

There are graph representations, and there are dynamic "program trace representations" which perform the correct analysis and computation during runtime (by i.e. collecting information specific to that execution).

\hrulefill

A static graph representation is amenable to analysis (like a compiler) but programming to that representation is more restrictive. This is often required to achieve the absolute best performance (see XLA).

\hrulefill

The dynamic representation usually allows you to program directly in the host language.

#

There is no single "best" way to combine deep learning and PP...

\begin{itemize}
\item Sampling from a distribution is not directly differentiable [1]. When including sampling, the correct thing is to compute the expected loss over the sampling distribution.
\item Static, graph based representations of probabilistic programs allow for restricted use of "black box" operations (i.e. I want to shove a deep network in the middle of my PP). 

This then restricts what inference algorithms you can use.
\item How do you differentiate a probabilistic program [2]? Can you apply reparametrization tricks to a PP?
\end{itemize}

\hrulefill

[1] This requires the usage of _reparametrization tricks_ which produce gradient estimators.

[2] Semantically, this is understood as: you differentiate the log probability of the data. But this is not trivial to acquire for all models.


# A modern tower of Babel

Can you really blame these communities (and the implementors of frameworks) for not talking to each other? 

It requires tremendous experience to become high-level in any one of these areas. Additionally, they care about performance in different ways:

\begin{itemize}
\item Deep learning: must deploy to heterogeneous accelerators and many GPUs.
\item PP: complex proposals require high CPU power - some exact algorithms are just matrix muls. More important in general to thus parallelize across many threads of a CPU.
\end{itemize}

\hrulefill

Let's take a closer look at the abstractions to identify the microcosm of the main issue: improvement to "mutual abstractions" is not a high priority.

# Automatic differentiation {.fragile}

```julia
function foo(x::Float64, y::Float64)
    return sin(exp(x) + y)
end
```

\hrulefill

Simple - but the core ideas apply to complex differentiable programs.

#

The chain rule of differential calculus
\begin{equation}
\frac{d}{dx}f(g(h(x))) = \big(\frac{df}{df}\biggr|_{f(x)}\big)\times\big(\frac{df}{dg}\biggr|_{g(x)}\big)\times\big(\frac{dg}{dh}\biggr|_{h(x)}\big)\times\big(\frac{dh}{dx}\biggr|_x\big)
\end{equation}
\hrulefill
\begin{itemize}
\item Forward mode: start on the right hand side.
\item Reverse mode: start on the left hand side.
\end{itemize}

# Forward mode

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{figure}
\includegraphics[width=5.5cm]{computational_graph.png}
\caption{A computation graph.}
\end{figure}
\end{column}
\begin{column}{0.5\textwidth}
\begin{align*}
x = x &\Longrightarrow dx = dx \\
y = y &\Longrightarrow dy = dy \\
y_1 = exp(x) &\Longrightarrow dy_1 = exp(x)~dx\\
y_2 = y_1 + y &\Longrightarrow dy_2 = dy_1 + dy \\
y_3 = sin(y_2) &\Longrightarrow dy_3 = cos(y_2)~dy_2\\
\end{align*}
\end{column}
\end{columns}

# Reverse mode

\begin{figure}
\includegraphics[width=5.5cm]{reverse_mode.png}
\caption{An adjoint computation graph.}
\end{figure}

#

The core representation is a "computation graph" which represents the flow of \textit{differentiable} information through the program.

\begin{columns}
\begin{column}{0.5\textwidth}
\begin{figure}
\includegraphics[width=5cm]{pytorch.png}
\end{figure}
\end{column}
\begin{column}{0.5\textwidth}
\begin{figure}
\includegraphics[width=5cm]{tensorflow.jpeg}
\end{figure}
\end{column}
\end{columns}

Depending on the implementation, this graph may be represented explicitly before execution (i.e. TF 1) or implicitly by execution (PyTorch and TF 2/Eager).

# Probabilistic programming

```julia
function foo(x::Float32)
    y = rand(Normal(x, 1.0))
    z = rand(Normal(y, 1.0))
    return z
end
```

\hrulefill

The key operation for probabilistic programming is _inference_. 

\begin{equation}
\tag{Bayes}
P(y | z) = \frac{P(z | y) P(y)}{P(z)}
\end{equation}

Given observed data $z$, I want to update the distribution over $y$ to reflect the data.

#

\hrulefill

| Forms of inference | Requires       |
|--------------------|----------------|
|Sampling algorithms | Ability to sample from program |
| | |
|Exact inference with conjugate families | Ability to reason about primitive distributions in program |
| | |
|Variational inference | In black box version, the ability to sample from program |
| | |
|Belief propagation | Ability to reason about primitive distributions in program |

# Static (graph-based) PPLs

See _Figaro_, for example.

Here, the model is constructed by explicitly connecting the graph together _before_ runtime.

\hrulefill

\begin{itemize}
\item Advantage: appears to be the closest to the computation graph abstraction. 
\item Advantage: a static representation allows analysis (as in every compiler ever).
\item Disadvantage: requires that you learn a DSL.
\item Disadvantage: Not necessarily compatible with other interpretations (i.e. AD) - depending on model, might be difficult to compute log prob.
\end{itemize}

# Trace-based PPLs

```julia
function foo(x::Float32)
    res = Array{Float64, 1}([])
    while rand(Normal(0.0, 1.0)) < 3.0
        push!(res, 1)
    end
    if length(res) > 10
        y = rand(Normal(10.0, 5.0))
    else
        y = rand(Normal(5.0, 3.0))
    end
    return y
end
```

\hrulefill

Key insight to understanding: running this program samples from a distribution over execution traces. What measurable space is that distribution defined on?

# The choice map abstraction {.fragile}

```julia
@gen function foo(x::Float32)
    res = Array{Float64, 1}([])
    counter = 0
    # :x => counter is addr
    while @trace(normal(0.0, 1.0),
                 :x => counter) < 1.0
        push!(res, 5)
        counter += 1
    end
    return res
end
```

We explicitly represent the trace by storing addresses and random choices.

#

Here's a sample choicemap:

\begin{figure}
\includegraphics[width=7cm]{example_trace.jpeg}
\end{figure}

\hrulefill

Not shown: accumulated log probabilities which go into the total log probability for this particular choice map.

# Monte Carlo sampling engines

\hrulefill

## Importance sampling
\begin{align*}
\mathbb{E}_{z\sim P(z | x)}\big[f(z)\big] &= \int f(z)~P(z | x)~dz \\
&= \int f(z)~\underbrace{\frac{P(z | x)}{Q(z)}~Q(z)}_{\text{Requires Q a.c. wrt P}}~dz \\
\tag{Monte Carlo estimate}
&\approx (\sum_{i=0}^Nw_i~f(z_i))/(\sum_{i=0}^N w_i)\\
\tag{Importance weights}
w_i &= \frac{P(x | z_i)P(z_i)}{Q(z_i)}
\end{align*}

# Programmable inference

In trace-based sampling methods, you write down a proposal \textit{program} $Q(z)$. 

The only constraint is that the program has to have the same support as the unobserved variables in the original one.

\hrulefill

Note that this follows from the absolute continuity requirement given on the previous slide.

#

\hrulefill

\begin{itemize}

\item Advantage: You can construct highly complex proposal distributions for sampling algorithms.

\item Advantage: Gradient information always available using trace-based auto diff.

\item Disadvantage: no static analysis - sampling always happens at runtime. You miss out on things like e.g. identifying conjugacy.

\item Disadvantage: numerous exact and approximate inference algorithms don't fit well into this framework (i.e. belief propagation based algorithms).

\item Disadvantage: auto diff $\textit{must be}$ trace-based. You can't do source-to-source AD if you only know the log probability at runtime.

\end{itemize}

# A summary of representation issues

Here's the crux - language issues exist between fields...but they also exist _within fields_!

\hrulefill

Deep learning:
\begin{itemize}
\item Limited support for complex architectures (e.g. TreeRNN, capsule networks, neural program policies) - i.e. the things we should be trying.
\item True $\textit{differentiable programming}$ where everything has an adjoint.
\end{itemize}

Probabilistic programming:
\begin{itemize}
\item Loss of useful inference information at runtime.
\item Static frameworks seen as "less flexible" compared to trace-based.
\end{itemize}

#

These issues have always existed between static and dynamic systems. 

They arise from the fundamental tension between what can be determined by the compiler and what cannot be determined until data is flowing.

\hrulefill

\centering
Machine learning people $\neq$ compiler people

...

but they are starting to learn about compilers.

# Combining 'contexts'

What about AD and probabilistic programming all in one?

\hrulefill

\begin{itemize}
\item This is a language problem.
\begin{enumerate}
\item What subset of your language supports probabilistic programming? What subset of your language is differentiable? Can they be made to intersect?
\end{enumerate}
\item Design space:
\begin{enumerate}
\item Use trace based for everything? Poor performance on complex architectures. Think: pure interpreter approach.
\item Source-to-source? No existing source-to-source PPL package which is ``universal" - initial work done on "density compilers" (see $\textit{Hakaru}$). Ongoing work on $\textit{Jaynes}$.
\end{enumerate}
\end{itemize}

#

\begin{figure}
\includegraphics[width=10cm]{intersection.png}
\end{figure}

# The future...

Many bleeding edge efforts restrict themselves to a purely functional subset of a host language...

\begin{itemize}
\item JAX - functional subset of Python (yikes!)
\item NumPyro - effect handlers in Python (yikes!)
\item Possibly XLA... (not in Python)
\end{itemize}

Pure functional languages are easy to reason about because mutability is controlled and composition makes flow-of-control very explicit.

In other words, everything is local...can we make AD + PP local as well?

# Algebraic effects

A principled way to write programs with "semantic interception".

\hrulefill

If you've ever used exception handling, you've used an effect.

Let's say you write some code, and you want to imbue it with a "context" where certain calls are handled in a particular way.

#

\begin{figure}
\includegraphics[width=9cm]{effects_AD.jpeg}
\end{figure}

#
So a program gets handled in a context...the handler can express many forms of computation.

\hrulefill

\begin{figure}
\includegraphics[width=6cm]{effect_handling.jpeg}
\caption{More explicit: any call which requires the $\textit{Differentiable}$ ability in the context is handled explicitly by a specific handler.}
\end{figure}

#

Why restrict this to AD? Why not PP?

\hrulefill

(I'm glad you asked...)

#

\begin{figure}
\includegraphics[width=9cm]{pp.jpeg}
\end{figure}

#

\begin{figure}
\includegraphics[width=9cm]{res_pp.jpeg}
\end{figure}

#

Why not both _at the same time_?

\hrulefill

(Again, wonderful question inquisitive listener...)

# 

Effects are a really wonderful idea. 

\hrulefill
\begin{itemize}
\item Functional way (i.e. composable!) way to handle side effects.
\item Can be type checked!
\end{itemize}

They allow you to explicitly control and check for _context_ with the type system!

\hrulefill

In _Scruff_ development, we have a notion called _model capabilities_ which I believe is modelled by functional effect systems.

# Summary

At their cores, deep learning and probabilistic programming systems are very similar. However, what information is required to compute the thing of interest, as well as how it is computed are very different.

\hrulefill

An interesting avenue for research: unified representations for machine learning systems which allow the flexibility to express complex differentiable models with probabilistic ones.
