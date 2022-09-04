# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Motjaba Sadegh, and Alexander Kuhn-Regnier
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import random
import time
from warnings import warn

import numpy as np
from colorama import Back, Fore, Style
from colorama import init as colorama_init
from python_inferno.cache import mark_dependency

from . import _algorithm


def format_number(x, format_str, prev, back="", comp_direction="+"):
    if comp_direction == "+":
        comp_colours = (Fore.GREEN, Fore.RED)
    elif comp_direction == "-":
        comp_colours = (Fore.RED, Fore.GREEN)
    else:
        raise ValueError(comp_direction)

    if prev is None:
        c = ""
    elif x > prev:
        c = comp_colours[0]
    elif x == prev:
        c = Fore.YELLOW
    else:
        c = comp_colours[1]

    fmt_x = format(x, format_str)

    return f"{back}{c}{fmt_x}{Style.RESET_ALL}"


class dream(_algorithm):
    """
    Implements the DiffeRential Evolution Adaptive Metropolis (DREAM) algorithhm
    based on:
    Vrugt, J. A. (2016) Markov chain Monte Carlo simulation using the DREAM software package.
    """

    @mark_dependency
    def __init__(self, *args, **kwargs):
        """
        Input
        ----------
        spot_setup: class
            model: function
                Should be callable with a parameter combination of the parameter-function
                and return an list of simulation results (as long as evaluation list)
            parameter: function
                When called, it should return a random parameter combination. Which can
                be e.g. uniform or Gaussian
            objectivefunction: function
                Should return the objectivefunction for a given list of a model simulation and
                observation.
            evaluation: function
                Should return the true values as return by the model.

        dbname: str
            * Name of the database where parameter, objectivefunction value and simulation results will be saved.

        dbformat: str
            * ram: fast suited for short sampling time. no file will be created and results are saved in an array.
            * csv: A csv file will be created, which you can import afterwards.

        parallel: str
            * seq: Sequentiel sampling (default): Normal iterations on one core of your cpu.
            * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).

        save_sim: boolean
            * True:  Simulation results will be saved
            * False: Simulation results will not be saved
        """

        kwargs["optimization_direction"] = "maximize"
        kwargs[
            "algorithm_name"
        ] = "DiffeRential Evolution Adaptive Metropolis (DREAM) algorithm"
        super(dream, self).__init__(*args, **kwargs)
        self._print_init()

    def _print_init(self):
        # colorama_init()  # Interferes with e.g. 'tee' command.
        self.print_status_s = 10  # Time between status updates.

        self.print_width = 100  # Nr. of characters.

        self.last_print_accept_rates = None
        self.last_print_eps_perc = None
        self.last_print_r_hat = None
        self.last_print_pCR = None
        self.last_print_chain_move_counts = None

    def format_status_header(self, s, fill="-"):
        to_fill = self.print_width - len(s) - 2
        left_fill = math.floor(to_fill / 2)
        right_fill = math.ceil(to_fill / 2)
        return fill * left_fill + f" {s} " + fill * right_fill

    def conc_status_elements(self, elements, n):
        """Concatenate a collection of strings across rows.

        Args:
            elements (iterable of str): Elements to concatenate.
            n (int): Length of each string (excluding characters which will not
                physically take up space like colour codes). This is constant across
                all elements.

        """
        S = len(elements)

        # Take into account 1 space padding between all elements.
        per_row = math.floor((self.print_width + 1) / (n + 1))

        math.ceil(S / per_row)

        rows = []

        for lower, upper in zip(
            range(0, S, per_row), range(per_row, S + per_row, per_row)
        ):
            # 1 space padding between elements.
            rows.append(" ".join(elements[lower:upper]))

        return "\n".join(rows)

    @mark_dependency
    def check_par_validity_bound(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            raise ValueError(
                "ERROR: Bounds do not have the same lengths as Parameterarray"
            )
        return par

    @mark_dependency
    def get_regular_startingpoint(self, nChains):
        randompar = self.parameter()["random"]
        for i in range(1000):
            randompar = np.column_stack((randompar, self.parameter()["random"]))
        startpoints = []
        for j in range(nChains):
            startpoints.append(
                np.percentile(randompar, (j + 1) / float(nChains + 1) * 100, axis=1)
            )  # ,np.amax(randompar,axis=1)
        startpoints = np.array(startpoints)
        for k in range(len(randompar)):
            random.shuffle(startpoints[:, k])
        return startpoints

    @mark_dependency
    def check_par_validity_reflect(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i] + (self.min_bound[i] - par[i])
                elif par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i] - (par[i] - self.max_bound[i])

            # Postprocessing if reflecting jumped out of bounds
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            raise ValueError(
                "ERROR: Bounds do not have the same lengths as Parameterarray"
            )
        return par

    @mark_dependency
    def _get_gamma(self, newN, nchain_pairs):
        # N = Number of parameters
        p = np.random.uniform(low=0, high=1)
        if p >= 0.2:
            # d_star is the dimension of subspace of parameters to jump
            d_star = np.sum(newN)
            gamma = 2.38 * self.beta / np.sqrt(2 * nchain_pairs * d_star)
        else:
            gamma = 1
        return gamma

    @mark_dependency
    def get_other_random_chains(self, cur_chain, nchain_pairs):
        chain_pairs = []
        selectable_chain = list(range(self.nChains))
        selectable_chain.remove(cur_chain)

        for i in range(nchain_pairs):
            pair_ith = random.sample(selectable_chain, 2)
            chain_pairs.append(pair_ith)
            for chain in pair_ith:
                selectable_chain.remove(chain)

        return chain_pairs

    @mark_dependency
    def get_new_proposal_vector(self, cur_chain, newN):
        nchain_pairs = random.randint(1, self.delta)
        gamma = self._get_gamma(newN, nchain_pairs)
        chain_pairs = self.get_other_random_chains(cur_chain, nchain_pairs)
        cur_par_set = self.bestpar[cur_chain, self.nChainruns[cur_chain] - 1]

        new_parameterset = []
        random_par_sets1 = []  # contain all random_par_set1
        random_par_sets2 = []  # contain all random_par_set2

        for i in range(nchain_pairs):
            random_chain1 = chain_pairs[i][0]
            random_chain2 = chain_pairs[i][1]
            random_par_set1 = list(
                self.bestpar[random_chain1][self.nChainruns[random_chain1] - 1]
            )
            random_par_set2 = list(
                self.bestpar[random_chain2][self.nChainruns[random_chain2] - 1]
            )
            random_par_sets1.append(random_par_set1)
            random_par_sets2.append(random_par_set2)

        random_par_set1 = [
            sum(i) for i in zip(*random_par_sets1)
        ]  # sum all random_par_set1
        random_par_set2 = [
            sum(i) for i in zip(*random_par_sets2)
        ]  # sum all random_par_set2

        lambda_ = np.random.uniform(-self.c, self.c)

        for i in range(self.N):  # Go through parameters
            if newN[i]:
                new_parameterset.append(
                    cur_par_set[i]
                    + (1.0 + lambda_)
                    * gamma
                    * np.asarray(random_par_set1[i] - random_par_set2[i])
                    + np.random.normal(0, self.eps)
                )
            else:
                new_parameterset.append(cur_par_set[i])

        new_parameter = self.check_par_validity_reflect(new_parameterset)
        # new_parameter=self.check_par_validity_bound(new_parameterset)

        jump_vec = new_parameter - cur_par_set

        return new_parameter, jump_vec

    @mark_dependency
    def update_last_half_param_data_const_n(
        self,
        *,
        old_params,
        new_params,
        count,
        chain_sel=None,
    ):
        """Update r-hat data for constant n (parameter removal + insertion).

        chain_sel (int, slice, or None): Can be used to selectively update data for
            specific chains. By default, all chains are updated. Shape of `old_params`
            and `new_params` depends on given `chain_sel`.

        """
        if chain_sel is None:
            chain_sel = slice(None)

        old_means = self.mean_last_half_params[chain_sel].copy()

        assert old_means.shape == old_params.shape == new_params.shape

        # Update means.
        self.mean_last_half_params[chain_sel] += (new_params - old_params) / count

        new_means = self.mean_last_half_params[chain_sel]

        self.last_half_params_m2[chain_sel] += (
            (new_params - new_means) ** 2
            - (old_params - old_means) ** 2
            + (old_means - new_means)
            * ((count + 1) * new_means - 2 * new_params - (count - 1) * old_means)
        )

    @mark_dependency
    def update_last_half_param_data_inc_n(
        self,
        *,
        new_params,
        count,
    ):
        """Update r-hat data for increasing n (+1)."""
        # Using old means.
        delta = new_params - self.mean_last_half_params

        # Update means.
        self.mean_last_half_params += (new_params - self.mean_last_half_params) / count

        # Using new means.
        delta2 = new_params - self.mean_last_half_params

        self.last_half_params_m2 += delta * delta2

    @mark_dependency
    def update_r_hat_data(self):
        """Online calculation of mean and variance using Welford's method and a
        similarly derived algorithm for the case of parameter removal + insertion
        (constant number).

        Needs to be called ONCE after every update to parameters across all chains.

        """
        assert np.unique(self.nChainruns).size == 1
        n_valid = self.nChainruns[0]
        new_index = n_valid - 1

        # Latest parameters across all chains (chains, parameters).
        new_params = self.bestpar[:, new_index]
        count = math.ceil(n_valid / 2)
        if (n_valid % 2) == 0:
            # Addition of new parameters AND removal of old parameters (since we are
            # only interested in the last 1/2 of parameters).

            # NOTE Number of elements constant compared to last cycle.
            self.update_last_half_param_data_const_n(
                old_params=self.bestpar[:, new_index // 2],
                new_params=new_params,
                count=count,
            )
        else:
            # Only addition of new parameters (increase in the number of elements that
            # are part of the last 1/2).

            # NOTE Number of elements has increases by 1 compared to last cycle.
            self.update_last_half_param_data_inc_n(new_params=new_params, count=count)

    @mark_dependency
    def update_last_half_like(
        self, *, chain_index, chain_run_index, n_half, like, delta
    ):
        """Calculate the mean of the last half of all samples. Avoid repeatedly
        calling `mean()` on the increasingly filled array by using an online algorithm
        which stores a sum that is updated as needed."""
        self.bestlike_arr[chain_index, chain_run_index] = like

        # Calculate mean. Divide by number of elements currently used to compute it.

        # NOTE The number of elements that are supposed to used to compute the mean is
        # not verified in any way and thus relies on adherence to convention (i.e.
        # dependence on number of runs).

        self.mean_last_half_like[chain_index] += delta / n_half

    @mark_dependency
    def update_mcmc_status(self, par, like, sim, cur_chain):
        # NOTE As this is called, the number of runs has not been incremented yet,
        # i.e. there will be 1 computed element for run index 0, 2 elements for run
        # index 1, etc...
        chain_run_index = self.nChainruns[cur_chain]
        n_chain_runs = chain_run_index + 1

        self.bestpar[cur_chain, chain_run_index] = list(par)
        self.bestlike[cur_chain] = like
        self.bestsim[cur_chain] = list(sim)

        # Prepare `check()` data.
        self.update_last_half_like(
            chain_index=cur_chain,
            chain_run_index=chain_run_index,
            n_half=math.ceil(n_chain_runs / 2),
            like=like,
            delta=(
                # Add latest element
                like
                # AND remove earliest element from mean.
                - self.bestlike_arr[cur_chain, chain_run_index // 2]
                if ((n_chain_runs % 2) == 0)
                else
                # Only add latest element to mean.
                like - self.mean_last_half_like[cur_chain]
            ),
        )

    @mark_dependency
    def check(self):
        """'check' and removal of aberrant trajectories / dissident chains using IQR.

        See https://www.degruyter.com/document/doi/10.1515/IJNSNS.2009.10.3.273/html

        """
        assert np.unique(self.nChainruns).size == 1

        # NOTE Correct run index by `-1` (compared to `check()`) since it will
        # have been updated before `check()` is called in the main `sample()`
        # function.
        n_runs = self.nChainruns[0]  # Same across all chains.
        run_index = n_runs - 1

        q1, q3 = np.quantile(self.mean_last_half_like, [0.25, 0.75])
        min_thres = q1 - 2 * (q3 - q1)

        dissident_indices = np.where(self.mean_last_half_like < min_thres)[0]

        if np.any(dissident_indices):
            # Determine current best position (chain).
            best_chain_index = np.argmax(self.bestlike)

            best_chain_par = self.bestpar[best_chain_index, run_index]
            best_chain_like = self.bestlike[best_chain_index]
            best_chain_sim = self.bestsim[best_chain_index]

            for chain_index in dissident_indices:
                # Move this chain to the current best position.

                self.chain_move_counts[chain_index] += 1

                prev_like = self.bestlike[chain_index]
                prev_par = self.bestpar[chain_index, run_index].copy()

                self.bestpar[chain_index, run_index] = best_chain_par
                self.bestlike[chain_index] = best_chain_like
                self.bestsim[chain_index] = best_chain_sim

                n_half = math.ceil(n_runs / 2)

                # Update check data.
                self.update_last_half_like(
                    chain_index=chain_index,
                    chain_run_index=run_index,
                    n_half=n_half,
                    # Set new value.
                    like=best_chain_like,
                    # Update sum (mean) accordingly.
                    delta=best_chain_like - prev_like,
                )

                self.update_last_half_param_data_const_n(
                    old_params=prev_par,
                    new_params=best_chain_par,
                    count=n_half,
                    chain_sel=chain_index,
                )

    @mark_dependency
    def get_r_hat(self):
        """Get r-hat."""
        # NOTE shape is: (chains, repetitions, parameters).
        assert self.nChains > 3

        assert np.unique(self.nChainruns).size == 1
        n_valid = self.nChainruns[0]

        if n_valid < 3:
            # Need at least 3 elements to have 2 elements in the last half of the
            # chains.
            return np.zeros(self.nr_of_pars) + np.nan

        n_half = math.ceil(n_valid / 2)

        # Select last 1/2 of valid data.
        parameter_array = self.bestpar[:, math.floor(n_valid / 2) : n_valid, :]

        # Variance of each chain and parameter across all repetitions.
        # chain_var = np.var(parameter_array, axis=1, ddof=1)
        chain_var = self.last_half_params_m2 / (n_half - 1)

        # Mean of chains across repetitions for each parameter.
        # chain_means = np.mean(parameter_array, axis=1)
        chain_means = self.mean_last_half_params

        # Mean of chain variances for each parameter.
        W = np.mean(chain_var, axis=0)
        # Variance of chain means for each parameter.
        chain_means_var = np.var(chain_means, axis=0, ddof=1)
        # Between-chain variance for each parameter.
        B = n_half * chain_means_var
        # Weighted average of within-chain and between-chain variances.
        var_hat = ((n_half - 1) / n_half) * W + B / n_half
        R_hat = np.sqrt(var_hat / W)

        return R_hat

    def print_status(self):
        # Overall acceptance rates

        print(self.format_status_header(f"Acceptance rates [%]  |  t = {self.iter}"))

        accept_rates = 100 * self.accepted / self.iter

        # Special colour for the lowest acceptance rate.
        colours = [""] * len(accept_rates)
        colours[np.argmin(accept_rates)] = Back.WHITE

        fmt_accept_rates = [
            format_number(rate, ">7.2f", prev_rate, back=colour)
            for rate, prev_rate, colour in zip(
                accept_rates,
                self.last_print_accept_rates
                if self.last_print_accept_rates is not None
                else [None] * len(accept_rates),
                colours,
            )
        ]
        self.last_print_accept_rates = accept_rates

        print(self.conc_status_elements(fmt_accept_rates, n=7))

        # r-hat

        r_hat_vals = self.r_hats[-1]

        # Special colour for the biggest (i.e. worst) r-hat value.
        colours = [""] * len(r_hat_vals)
        colours[np.argmax(r_hat_vals)] = Back.WHITE

        fmt_r_hat = [
            # Change preferred direction - we want r-hat to decrease.
            format_number(r_hat, ">7.3f", prev_r_hat, back=colour, comp_direction="-")
            for r_hat, prev_r_hat, colour in zip(
                r_hat_vals,
                self.last_print_r_hat
                if self.last_print_r_hat is not None
                else [None] * len(r_hat_vals),
                colours,
            )
        ]
        self.last_print_r_hat = r_hat_vals

        print(self.format_status_header("r-hat"))
        print(self.conc_status_elements(fmt_r_hat, n=7))

        # Chain move counts

        # Highlight chains that have been moved recently.
        colours = [""] * self.nChains
        if self.last_print_chain_move_counts is not None:
            diff = self.chain_move_counts - self.last_print_chain_move_counts
            for index in np.where(diff > 0)[0]:
                colours[index] = Back.WHITE

        fmt_counts = [
            # Change preferred direction - chain moves are not ideal.
            format_number(count, ">7d", prev_count, back=colour, comp_direction="-")
            for count, prev_count, colour in zip(
                self.chain_move_counts,
                self.last_print_chain_move_counts
                if self.last_print_chain_move_counts is not None
                else [None] * self.nChains,
                colours,
            )
        ]
        self.last_print_chain_move_counts = self.chain_move_counts.copy()

        print(self.format_status_header("Total chain moves"))
        print(self.conc_status_elements(fmt_counts, n=7))

        # Acceptance tolerance (actually printed lower).

        eps_perc = 100 * self.acc_eps_stats[0] / np.sum(self.acc_eps_stats)
        fmt_eps_perc = format_number(eps_perc, ">8.5f", self.last_print_eps_perc)
        self.last_print_eps_perc = eps_perc

        # pCR

        fmt_pCR = " ".join(
            format_number(pCR_i, "0.4f", prev_pCR_i)
            for pCR_i, prev_pCR_i in zip(
                self.pCR,
                self.last_print_pCR
                if self.last_print_pCR is not None
                else [None] * len(self.pCR),
            )
        )
        self.last_print_pCR = self.pCR

        print(Style.DIM + "-" * self.print_width + Style.RESET_ALL)
        print(
            "pCR = "
            + fmt_pCR
            + "  |  "
            + f"better than eps ({self.acc_eps:0.7f}) = {fmt_eps_perc}%"
        )
        print("-" * self.print_width)

    @mark_dependency
    def sample(
        self,
        repetitions,
        nChains=7,
        nCr=3,
        delta=3,
        c=0.1,
        eps=10e-6,
        convergence_limit=1.2,
        runs_after_convergence=1000,
        beta=1.0,
        maxTime=np.inf,
        burnInNSamples=None,
        acc_eps=0.05,
    ):
        """DREAM(ABC) sampling.

        beta ([0.0, 1.0]): Scales jump distance (except for when gamma=1). Decreasing
            `beta` can increase acceptance rates.
        maxTime (float): Maximum time in seconds.
        burnInNSamples (int or None): During the first `burnInNSamples` samples,
            aberrant chain correction using the IQR measure will be used. If None,
            will be set to `repetitions / 2`.
        acc_eps (float): Acceptance tolerance (see DREAM(ABC) algorithm). Chosen
            heuristically. Should be reasonably small without excessively decreasing
            acceptance ratios. Magnitude of `acc_eps` is related to the provided loss
            function. Increasing `acc_eps` can increase acceptance rates.

        """
        if nChains < 2 * delta + 1:
            print("Please use at least n=2*delta+1 chains!")
            return None

        # Prepare storing MCMC chain as array of arrays.
        # Define stepsize of MCMC.
        self.repetitions = int(repetitions)
        self.nChains = int(nChains)
        self.delta = delta

        self.acc_eps = acc_eps

        # NOTE The internal status object counts total repetitions
        # (i.e. repetitions * chains).
        self.set_repetition(self.repetitions * self.nChains)

        self.burnInNSamples = (
            int(burnInNSamples) if burnInNSamples is not None else self.repetitions // 2
        )
        self.beta = beta
        self.c = c

        self.stepsizes = self.parameter()["step"]  # array of stepsizes
        self.nr_of_pars = len(self.stepsizes)

        starttime = time.time()
        intervaltime = starttime

        # Metropolis-Hastings iterations.

        self.bestpar = (
            np.zeros((self.nChains, self.repetitions, self.nr_of_pars)) + np.nan
        )

        self.bestlike = [[-np.inf]] * self.nChains
        self.bestsim = [[np.nan]] * self.nChains
        self.accepted = np.zeros(self.nChains)
        self.nChainruns = [0] * self.nChains
        self.min_bound, self.max_bound = (
            self.parameter()["minbound"],
            self.parameter()["maxbound"],
        )

        # Dissident chain check data.
        self.bestlike_arr = np.zeros((self.nChains, self.repetitions)) + np.nan
        self.mean_last_half_like = np.zeros((self.nChains,))

        # r-hat online statistics data.

        self.mean_last_half_params = np.zeros((self.nChains, self.nr_of_pars))
        # Sum of squares (M2) which can be used to calculate the variance.
        self.last_half_params_m2 = np.zeros((self.nChains, self.nr_of_pars))

        self.iter = 0

        startpoints = self.get_regular_startingpoint(nChains)

        param_generator = (
            (cChain, list(startpoints[cChain])) for cChain in range(int(self.nChains))
        )  # TODO: Start with regular interval raster
        for cChain, par, sim in self.repeat(param_generator):
            like = self.postprocessing(par, sim, chains=cChain)
            self.update_mcmc_status(par, like, sim, cChain)
            self.nChainruns[cChain] += 1

        self.iter += 1

        self.update_r_hat_data()

        self.convergence = False

        self.r_hats = []
        self.eps = eps
        self.N = len(self.parameter()["random"])

        # Crossover values.
        self.CR = [(i + 1) / nCr for i in range(nCr)]
        # Initial crossover selection probabilities.
        self.pCR = np.ones(nCr) / nCr
        # Used to update crossover probabilities.
        self.J = np.zeros(nCr)
        self.n_id = np.zeros(nCr)
        # Store which crossover id was used to generate proposals.
        crossover_ids = np.zeros(self.nChains, dtype=np.int64)
        # Jump vector.
        dX = np.zeros((self.nChains, self.nr_of_pars))

        # Record how many samples are above / below the threshold.
        self.acc_eps_stats = np.zeros(2, dtype=np.int64)

        # Chain move stats.
        self.chain_move_counts = np.zeros(self.nChains, dtype=np.int64)

        while self.iter < self.repetitions:
            # Generate proposals for each chain.

            proposals = []

            # Standard deviation of each parameter at current iteration.
            std_X = np.std(self.bestpar[:, self.iter - 1], axis=0)

            for cChain in range(self.nChains):
                # Use crossover probabilities to select the crossover index.
                crossover_id = np.random.choice(a=nCr, p=self.pCR)
                crossover_ids[cChain] = crossover_id  # Store for later use.
                chosen_crossover = self.CR[crossover_id]  # Get crossover value.

                uniform_sample = np.random.uniform(low=0, high=1, size=self.N)
                newN = uniform_sample < chosen_crossover
                if not np.sum(newN):
                    # Assign single random element to be changed.
                    newN[np.random.randint(0, self.N)] = True

                # Get new parameter proposal.
                new_params, jump_vec = self.get_new_proposal_vector(cChain, newN)
                proposals.append((cChain, new_params))

                # Store jump vector.
                dX[cChain] = jump_vec

            # Evaluate and process each proposal.

            for cChain, par, sim in self.repeat(proposals):
                like = self.postprocessing(par, sim, chains=cChain)

                # Acceptance test based on DREAM(ABC) algorithm.
                # Acceptance tolerance `self.acc_eps` is user-provided and chosen heuristically.
                # Proposal is accepted if:
                # - New likelihood is higher.
                # - OR New likelihood exceeds given tolerance.
                #
                # See:
                # - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR015386
                # - https://hess.copernicus.org/articles/17/4831/2013/hess-17-4831-2013.pdf

                if like > self.bestlike[cChain] or (-like < self.acc_eps):
                    # Accepted.
                    self.update_mcmc_status(par, like, sim, cChain)
                    self.accepted[cChain] += 1  # monitor acceptance
                else:
                    # Rejected. Keep last state.
                    self.update_mcmc_status(
                        self.bestpar[cChain][self.nChainruns[cChain] - 1],
                        self.bestlike[cChain],
                        self.bestsim[cChain],
                        cChain,
                    )

                    # Reset jump vector.
                    dX[cChain] = 0

                if -like < self.acc_eps:
                    self.acc_eps_stats[0] += 1
                else:
                    self.acc_eps_stats[1] += 1

                # Update crossover stats.
                self.J[crossover_ids[cChain]] += np.sum((dX[cChain] / std_X) ** 2)
                self.n_id[crossover_ids[cChain]] += 1

                self.nChainruns[cChain] += 1

            self.iter += 1

            self.update_r_hat_data()

            r_hat = self.get_r_hat()
            self.r_hats.append(r_hat)

            if self.iter <= self.burnInNSamples:
                # If still in burnin.

                # Update crossover probabilities.
                self.pCR = self.J / self.n_id
                self.pCR /= np.sum(self.pCR)  # Normalise.

                # 'check' and removal of aberrant trajectories / dissident chains
                # using IQR.
                self.check()

            # Refresh progressbar.

            acttime = time.time()

            if (
                acttime - intervaltime >= self.print_status_s
                and self.iter >= 2
                and self.nChainruns[-1] >= 3
            ):
                self.print_status()
                intervaltime = time.time()

            if (
                self.iter > self.burnInNSamples
                and np.all(np.asarray(r_hat) < convergence_limit)
                and not self.convergence
                and self.nChainruns[-1] >= 5
            ):
                # Stop sampling

                # Calculate how many additional samples we can sample.
                if (self.repetitions - self.iter) < runs_after_convergence:
                    actual_runs_after_convergence = self.repetitions - self.iter
                    warn("Cannot sample as many runs after convergence as wanted!")
                else:
                    actual_runs_after_convergence = runs_after_convergence

                print("#############")
                print(
                    "Convergence has been achieved after "
                    + str(self.iter)
                    + " of "
                    + str(self.repetitions)
                    + " runs! Finally, "
                    + str(actual_runs_after_convergence)
                    + " runs will be additionally sampled to form the posterior distribution"
                )
                print("#############")
                self.repetitions = self.iter + actual_runs_after_convergence
                self.set_repetition(self.repetitions * self.nChains)
                self.convergence = True

            # Maximum time reached.
            if (acttime - starttime) >= maxTime:
                # Stop sampling.
                self.repetitions = self.iter

        self.print_status()
        self.final_call()

        return self.r_hats
