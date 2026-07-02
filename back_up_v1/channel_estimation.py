#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Functions related to OFDM channel estimation"""

import tensorflow as tf
import numpy as np
from scipy.special import jv
import itertools
from abc import abstractmethod
import json
from importlib_resources import files

from sionna.phy import config, dtypes, PI, SPEED_OF_LIGHT
from sionna.phy.block import Object, Block
from sionna.phy.channel.tr38901 import models
from sionna.phy.utils import flatten_last_dims, expand_to_rank
from sionna.phy.ofdm import ResourceGrid, RemoveNulledSubcarriers

class BaseChannelEstimator(Block):
    # pylint: disable=line-too-long
    r"""
    Abstract block for implementing an OFDM channel estimator

    Any block that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.phy.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.phy.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resouce elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        Resource grid

    interpolation_type : "nn" (default) | "lin" | "lin_time_avg"
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.phy.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).

    interpolator : `None` (default) | :class:`~sionna.phy.ofdm.BaseChannelInterpolator`
        An instance of ,
        such as :class:`~sionna.phy.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], `tf.complex`
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `tf.float`
        Variance of the AWGN

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, `tf.float`
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, resource_grid, interpolation_type="nn", interpolator=None, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

        assert isinstance(resource_grid, ResourceGrid),\
            "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._remove_nulled_scs = RemoveNulledSubcarriers(resource_grid,
                                                          precision=self.precision)
        # breakpoint()
        assert interpolation_type in ["nn","lin","lin_time_avg","mmse_dp",None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type
        # breakpoint()
        if interpolator is not None:
            # breakpoint()
            assert isinstance(interpolator, BaseChannelInterpolator), \
        "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            # breakpoint()
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern,
                                                time_avg=True)
        elif self._interpolation_type == "mmse_dp":
            # breakpoint()
            self._interpol = IndustrialLMMSEInterpolator(
                pilot_pattern=self._pilot_pattern,
                fft_size=resource_grid.fft_size,
                cp_length=resource_grid.cyclic_prefix_length,
                modulation_type=resource_grid.num_bits_per_symbol,
                pilot_sym_idx = resource_grid._pilot_ofdm_symbol_indices,
                scs = resource_grid.subcarrier_spacing,
                num_ofdm_symbols=resource_grid._num_ofdm_symbols,
                precision=precision
            )

        
            
            
        
                                    

        # elif self._interpolation_type  == "lmmse":
        #     self._interpol = LMMSEInterpolator(self._pilot_pattern)
        # breakpoint()
        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        # breakpoint()
        mask = flatten_last_dims(self._pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        # breakpoint()
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    @abstractmethod
    def estimate_at_pilot_locations(self, y_pilots, no):
        """
        Estimates the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implement this class.

        Input
        -----
        y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `tf.complex`
            Observed signals for the pilot-carrying resource elements

        no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `tf.float`
            Variance of the AWGN

        Output
        ------
        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `tf.complex`
            Channel estimates for the pilot-carrying resource elements

        err_var : Same shape as ``h_hat``, `tf.float`
            Channel estimation error variance for the pilot-carrying
            resource elements
        """
        pass

    def call(self, y, no):

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        # breakpoint()
        y_eff = self._remove_nulled_scs(y) #y.shape=(1, 1, 1, 14, 12)

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)#TensorShape([128, 1, 4, 1680])

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        # breakpoint()
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)#TensorShape([1, 1, 1, 1, 1, 3])
        #这里提取的导频2层是完全相同的，没有进行层区分
        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        # breakpoint()
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)
        # breakpoint()
        #y_pilots.shape=[1, 1, 16batch, 4tx, 2layer, 768])
        #h_hat.shape = y_pilots.shape
        # breakpoint()
        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            # breakpoint()
            h_hat, err_var = self._interpol(h_hat, err_var)
            #h_hat.shape[1, 1, 16, 4, 2, 14, 192]
            #err_var.shape=TensorShape([1, 1, 1, 1, 2, 14, 120])
            # breakpoint()
            err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))

        return h_hat, err_var

class LSChannelEstimator(BaseChannelEstimator):
    # pylint: disable=line-too-long
    r"""
    Block implementing least-squares (LS) channel estimation for OFDM MIMO systems

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        Resource grid

    interpolation_type : "nn" (default) | "lin" | "lin_time_avg"
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.phy.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).

    interpolator : `None` (default) | :class:`~sionna.phy.ofdm.BaseChannelInterpolator`
        An instance of ,
        such as :class:`~sionna.phy.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], `tf.complex`
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `tf.float`
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, `tf.float`
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """
    def estimate_at_pilot_locations(self, y_pilots, no):

        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        # breakpoint()
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)
        
        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)

        return h_ls, err_var

class BaseChannelInterpolator(Object):
    # pylint: disable=line-too-long
    r"""
    Abstract class for implementing an OFDM channel interpolator

    Any class that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.phy.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, `tf.float`
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    @abstractmethod
    def __call__(self, h_hat, err_var):
        pass

class NearestNeighborInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""
    Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.phy.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    accross a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : :class:`~sionna.phy.ofdm.PilotPattern`
        Used pilot pattern

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, `tf.float`
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)
        for a in range(gather_ind.shape[0]): # For each pilot pattern...
            i_p, j_p = np.where(mask[a]) # ...determine the pilot indices

            for i in range(mask_shape[-2]): # Iterate over...
                for j in range(mask_shape[-1]): # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i-i_p) + np.abs(j-j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a])==0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        self._gather_ind = tf.reshape(gather_ind, mask_shape)

    def _interpolate(self, inputs):
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0)
        inputs = tf.transpose(inputs, perm)

        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        outputs = tf.transpose(outputs, perm)

        return outputs

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var

class LinearInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""
    Linear channel estimate interpolation on a resource grid

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided channel estimates and
    error variances through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.phy.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    Parameters
    ----------
    pilot_pattern : :class:`~sionna.phy.ofdm.PilotPattern`
        Used pilot pattern

    time_avg : `bool`, (default `False`)
        If enabled, measurements will be averaged across OFDM symbols
        (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, `tf.float`
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern, time_avg=False):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        self._time_avg = time_avg

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        # Linear interpolation works as follows:
        # We compute for each resource element (RE)
        # x_0 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the first channel measurement was taken
        # x_1 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the second channel measurement was taken
        # y_0 : The first channel estimate
        # y_1 : The second channel estimate
        # x   : The x-value (i.e., sub-carrier index or OFDM symbol)
        #
        # The linearly interpolated value y is then given as:
        # y = (x-x_0) * (y_1-y_0) / (x_1-x_0) + y_0
        #
        # The following code pre-computes various quantities and indices
        # that are needed to compute x_0, x_1, y_0, y_1, x for frequency- and
        # time-domain interpolation.

        ##
        ## Frequency-domain interpolation
        ##
        self._x_freq = tf.cast(expand_to_rank(tf.range(0, mask.shape[-1]),
                                              7,
                                              axis=0),
                               pilots.dtype)

        # Permutation indices to shift batch_dims last during gather
        self._perm_fwd_freq = tf.roll(tf.range(6), -3, 0)

        x_0_freq = np.zeros_like(mask, np.int32)
        x_1_freq = np.zeros_like(mask, np.int32)

        # Set REs of OFDM symbols without any pilot equal to -1 (dummy value)
        x_0_freq[np.sum(np.abs(z), axis=-1)==0] = -1
        x_1_freq[np.sum(np.abs(z), axis=-1)==0] = -1

        y_0_freq_ind = np.copy(x_0_freq) # Indices used to gather estimates
        y_1_freq_ind = np.copy(x_1_freq) # Indices used to gather estimates

        # For each stream
        for a in range(z.shape[0]):

            pilot_count = 0 # Counts the number of non-zero pilots

            # Indices of non-zero pilots within the pilots vector
            pilot_ind = np.where(np.abs(pilots[a]))[0]

            # Go through all OFDM symbols
            for i in range(x_0_freq.shape[1]):

                # Indices of non-zero pilots within the OFDM symbol
                pilot_ind_ofdm = np.where(np.abs(z[a][i]))[0]

                # If OFDM symbol contains only one non-zero pilot
                if len(pilot_ind_ofdm)==1:
                    # Set the indices of the first and second pilot to the same
                    # value for all REs of the OFDM symbol
                    x_0_freq[a][i] = pilot_ind_ofdm[0]
                    x_1_freq[a][i] = pilot_ind_ofdm[0]
                    y_0_freq_ind[a,i] = pilot_ind[pilot_count]
                    y_1_freq_ind[a,i] = pilot_ind[pilot_count]

                # If OFDM symbol contains two or more pilots
                elif len(pilot_ind_ofdm)>=2:
                    x0 = 0
                    x1 = 1

                    # Go through all resource elements of this OFDM symbol
                    for j in range(x_0_freq.shape[2]):
                        x_0_freq[a,i,j] = pilot_ind_ofdm[x0]
                        x_1_freq[a,i,j] = pilot_ind_ofdm[x1]
                        y_0_freq_ind[a,i,j] = pilot_ind[pilot_count + x0]
                        y_1_freq_ind[a,i,j] = pilot_ind[pilot_count + x1]
                        if j==pilot_ind_ofdm[x1] and x1<len(pilot_ind_ofdm)-1:
                            x0 = x1
                            x1 += 1

                pilot_count += len(pilot_ind_ofdm)

        x_0_freq = np.reshape(x_0_freq, mask_shape)
        x_1_freq = np.reshape(x_1_freq, mask_shape)
        x_0_freq = expand_to_rank(x_0_freq, 7, axis=0)
        x_1_freq = expand_to_rank(x_1_freq, 7, axis=0)
        self._x_0_freq = tf.cast(x_0_freq, pilots.dtype)
        self._x_1_freq = tf.cast(x_1_freq, pilots.dtype)

        # We add +1 here to shift all indices as the input will be padded
        # at the beginning with 0, (i.e., the dummy index -1 will become 0).
        self._y_0_freq_ind = np.reshape(y_0_freq_ind, mask_shape)+1
        self._y_1_freq_ind = np.reshape(y_1_freq_ind, mask_shape)+1

        ##
        ## Time-domain interpolation
        ##
        self._x_time = tf.expand_dims(tf.range(0, mask.shape[-2]), -1)
        self._x_time = tf.cast(expand_to_rank(self._x_time, 7, axis=0),
                               dtype=pilots.dtype)

        # Indices used to gather estimates
        self._perm_fwd_time = tf.roll(tf.range(7), -3, 0)

        y_0_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices

        # For each stream
        for a in range(z.shape[0]):

            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind)==1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind)>=2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a,i] = ofdm_ind[x0]
                    y_1_time_ind[a,i] = ofdm_ind[x1]
                    if i==ofdm_ind[x1] and x1<len(ofdm_ind)-1:
                        x0 = x1
                        x1 += 1

        self._y_0_time_ind = np.reshape(y_0_time_ind, mask_shape[:-1])
        self._y_1_time_ind = np.reshape(y_1_time_ind, mask_shape[:-1])

        self._x_0_time = expand_to_rank(tf.expand_dims(self._y_0_time_ind, -1),
                                                       7, axis=0)
        self._x_0_time = tf.cast(self._x_0_time, dtype=pilots.dtype)
        self._x_1_time = expand_to_rank(tf.expand_dims(self._y_1_time_ind, -1),
                                                       7, axis=0)
        self._x_1_time = tf.cast(self._x_1_time, dtype=pilots.dtype)

        #
        # Other precomputed values
        #
        # Undo permutation of batch_dims for gather
        self._perm_bwd = tf.roll(tf.range(7), 3, 0)

        # Padding for the inputs
        pad = np.zeros([6, 2], np.int32)
        pad[-1, 0] = 1
        self._pad = pad

        # Number of ofdm symbols carrying at least one pilot.
        # Used for time-averaging (optional)
        n = np.sum(np.abs(np.reshape(z, mask_shape)), axis=-1, keepdims=True)
        n = np.sum(n>0, axis=-2, keepdims=True)
        self._num_pilot_ofdm_symbols = expand_to_rank(n, 7, axis=0)

    def _interpolate_1d(self, inputs, x, x0, x1, y0_ind, y1_ind):
        # Gather the right values for y0 and y1
        y0 = tf.gather(inputs, y0_ind, axis=2, batch_dims=2)
        y1 = tf.gather(inputs, y1_ind, axis=2, batch_dims=2)

        # Undo the permutation of the inputs
        y0 = tf.transpose(y0, self._perm_bwd)
        y1 = tf.transpose(y1, self._perm_bwd)

        # Compute linear interpolation
        slope = tf.math.divide_no_nan(y1-y0, tf.cast(x1-x0, dtype=y0.dtype))
        return tf.cast(x-x0, dtype=y0.dtype)*slope + y0

    def _interpolate(self, inputs):
        #
        # Prepare inputs
        #
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Pad the inputs with a leading 0.
        # All undefined channel estimates will get this value.
        inputs = tf.pad(inputs, self._pad, constant_values=0)

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, 1+num_pilots, k, l, m]
        inputs = tf.transpose(inputs, self._perm_fwd_freq)

        #
        # Frequency-domain interpolation
        #
        # h_hat_freq has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_freq = self._interpolate_1d(inputs,
                                          self._x_freq,
                                          self._x_0_freq,
                                          self._x_1_freq,
                                          self._y_0_freq_ind,
                                          self._y_1_freq_ind)
        #
        # Time-domain interpolation
        #

        # Time-domain averaging (optional)
        if self._time_avg:
            num_ofdm_symbols = h_hat_freq.shape[-2]
            h_hat_freq = tf.reduce_sum(h_hat_freq, axis=-2, keepdims=True)
            h_hat_freq /= tf.cast(self._num_pilot_ofdm_symbols,h_hat_freq.dtype)
            h_hat_freq = tf.repeat(h_hat_freq, [num_ofdm_symbols], axis=-2)

        # Transpose h_hat_freq to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers, k, l, m]
        h_hat_time = tf.transpose(h_hat_freq, self._perm_fwd_time)

        # h_hat_time has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_time = self._interpolate_1d(h_hat_time,
                                          self._x_time,
                                          self._x_0_time,
                                          self._x_1_time,
                                          self._y_0_time_ind,
                                          self._y_1_time_ind)

        return h_hat_time

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)

        # the interpolator requires complex-valued inputs
        err_var = tf.cast(err_var, tf.complex64)
        # breakpoint()
        err_var = self._interpolate(err_var)#input(TensorShape([1, 1, 1, 1, 2, 120])),output(TensorShape([1, 1, 1, 1, 2, 14, 120]))
        err_var = tf.math.real(err_var)

        return h_hat, err_var

class LMMSEInterpolator1D(Object):
    # pylint: disable=line-too-long
    r"""
    Linear interpolation across the inner dimension of the input ``h_hat``

    The two inner dimensions of the input ``h_hat`` form a matrix :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`.
    LMMSE interpolation is performed across the inner dimension as follows:

    .. math::
        \tilde{\mathbf{h}}_n = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` interpolation LMMSE matrix:

    .. math::
        \mathbf{A}_n = \mathbf{R} \mathbf{\Pi}_n \left( \mathbf{\Pi}_n^\intercal \mathbf{R} \mathbf{\Pi}_n + \tilde{\mathbf{\Sigma}}_n \right)^{-1} \mathbf{\Pi}_n^\intercal.

    where :math:`\mathbf{R}` is the :math:`M \times M` covariance matrix across the inner dimension of the quantity which is estimated,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads :math:`K_n`
    values to a vector of size :math:`M` according to the ``pilot_mask`` for the :math:`n^{\text{th}}` row,
    and :math:`\tilde{\mathbf{\Sigma}}_n \in \mathbb{R}^{K_n \times K_n}` is the regularized channel estimation error covariance.
    The :math:`i^{\text{th}}`` diagonal element of :math:`\tilde{\mathbf{\Sigma}}_n` is such that:

    .. math::

        \left[ \tilde{\mathbf{\Sigma}}_n \right]_{i,i} = \text{max} \left\{  \right\}

     built from ``err_var`` and assumed to be diagonal.

    The returned channel estimates are

    .. math::
        \begin{bmatrix}
            {\tilde{\mathbf{h}}_1}^\intercal\\
            \vdots\\
            {\tilde{\mathbf{h}}_N}^\intercal
        \end{bmatrix}.

    The returned channel estimation error variances are the diaginal coefficients of

    .. math::
        \text{diag} \left( \mathbf{R} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R} \right), 1 \leq n \leq N

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size :math:`M \times M` that zeros the
    columns corresponding to rows not carrying any pilots.
    Note that interpolation is not performed for rows not carrying any pilots.

    **Remark**: The interpolation matrix differs across rows as different
    rows may carry pilots on different elements and/or have different
    estimation error variances.

    Parameters
    ----------
    pilot_mask : [:math:`N`, :math:`M`] : `int`
        Mask indicating the allocation of resource elements
        0 : Data,
        1 : Pilot,
        2 : Not used,

    cov_mat : [:math:`M`, :math:`M`], `tf.complex`
        Covariance matrix of the channel across the inner dimension

    last_step : `bool`
        Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `tf.complex`
        Channel estimates

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `tf.complex`
        Channel estimation error variances

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, :math:`N`, :math:`M`], `tf.complex`
        Channel estimates interpolated across the inner dimension

    err_var : Same shape as ``h_hat``, `tf.float`
        The channel estimation error variances of the interpolated channel estimates
    """
    def __init__(self, pilot_mask, cov_mat, last_step):

        if cov_mat.dtype==tf.complex64:
            precision = "single"
        elif cov_mat.dtype==tf.complex128:
            precision = "double"
        else:
            msg = "`cov_mat` dtype must be one of tf.complex64 or tf.complex128"
            raise TypeError(msg)
        super().__init__(precision=precision)

        self._rzero = tf.constant(0.0, self.rdtype)

        # Interpolation is performed along the inner dimension of
        # the resource grid, which may be either the subcarriers
        # or the OFDM symbols dimension.
        # This dimension is referred to as the inner dimension.
        # The other dimension of the resource grid is referred to
        # as the outer dimension.

        # Size of the inner dimension.
        inner_dim_size = tf.shape(pilot_mask)[-1]
        self._inner_dim_size = inner_dim_size

        # Size of the outer dimension.
        outer_dim_size = tf.shape(pilot_mask)[-2]
        self._outer_dim_size = outer_dim_size
        # breakpoint()
        self._cov_mat = cov_mat
        self._last_step = last_step

        # Computation of the interpolation matrix is done solving the
        # least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        # Extracting the locations of pilots from the pilot mask
        num_tx = tf.shape(pilot_mask)[0]
        num_streams_per_tx = tf.shape(pilot_mask)[1]

        # List of indices of pilots in the inner dimension for every
        # transmit antenna, stream, and outer dimension element.
        pilot_indices = []
        # Maximum number of pilots carried by an inner dimension.
        max_num_pil = 0
        # Indices used to add the error variance to the diagonal
        # elements of the covariance matrix restricted
        # to the elements carrying pilots.
        # These matrices are computed below.
        add_err_var_indices = np.zeros([num_tx, num_streams_per_tx,
                                        outer_dim_size, inner_dim_size, 5], int)
        for tx in range(num_tx):
            pilot_indices.append([])
            for st in range(num_streams_per_tx):
                pilot_indices[-1].append([])
                for oi in range(outer_dim_size):
                    pilot_indices[-1][-1].append([])
                    num_pil = 0 # Number of pilots on this outer dim
                    for ii in range(inner_dim_size):
                        # Check if this RE is carrying a pilot
                        # for this stream
                        if pilot_mask[tx,st,oi,ii] == 0:
                            continue
                        if pilot_mask[tx,st,oi,ii] == 1:
                            pilot_indices[tx][st][oi].append(ii)
                            indices = [tx, st, oi, num_pil, num_pil]
                            add_err_var_indices[tx, st, oi, ii] = indices
                            num_pil += 1
                    max_num_pil = max(max_num_pil, num_pil)
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, 5]
        # breakpoint()
        self._add_err_var_indices = tf.cast(add_err_var_indices, tf.int32)

        # Different subcarriers/symbols may carry a different number of pilots.
        # To handle such cases, we create a tensor of square matrices of
        # size the maximum number of pilots carried by an inner dimension
        # and zero-padding is used to handle axes with less pilots than the
        # maximum value. The obtained structure is:
        #
        # |B 0|
        # |0 0|
        #
        pil_cov_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                                max_num_pil, max_num_pil], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            tmp = np.take(cov_mat, pil_ind, axis=0)
            pil_cov_mat_ = np.take(tmp, pil_ind, axis=1)
            pil_cov_mat[tx,st,oi,:num_pil,:num_pil] = pil_cov_mat_
            #从协方差矩阵中提取仅导频位置的子矩阵，存入pilot_cov_mat中
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        # breakpoint()
        self._pil_cov_mat = tf.constant(pil_cov_mat, self.cdtype)

        # Pre-compute the covariance matrix with only the columns corresponding
        # to pilots.
        b_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                                max_num_pil, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            b_mat_ = np.take(cov_mat, pil_ind, axis=0)
            b_mat[tx,st,oi,:num_pil,:] = b_mat_
        self._b_mat = tf.constant(b_mat, self.cdtype)
        # breakpoint()
        # Indices used to fill with zeros the columns of the interpolation
        # matrix not corresponding to zeros.
        # The results is a matrix of size inner_dim_size x inner_dim_size
        # where rows and columns not correspondong to pilots are set to zero.
        pil_loc = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                            inner_dim_size, max_num_pil, 5], dtype=int)
        for tx,st,oi,p,ii in itertools.product(range(num_tx),
                                                range(num_streams_per_tx),
                                                range(outer_dim_size),
                                                range(max_num_pil),
                                                range(inner_dim_size)):
            if p >= len(pilot_indices[tx][st][oi]):
                # An extra dummy subcarrier is added to push there padding
                # identity matrix
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              inner_dim_size,
                                              inner_dim_size]
            else:
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              ii,
                                              pilot_indices[tx][st][oi][p]]
        self._pil_loc = tf.cast(pil_loc, tf.int32)

        # Covariance matrix for each stream with only the row corresponding
        # to a pilot carrying RE not set to 0.
        # This is required to compute the estimation error variances.
        err_var_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                inner_dim_size, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            mask = np.zeros([inner_dim_size], complex)
            mask[pil_ind] = 1.0
            mask = np.expand_dims(mask, axis=1)
            err_var_mat[tx,st,oi] = cov_mat*mask
        self._err_var_mat = tf.constant(err_var_mat, self.cdtype)

    def __call__(self, h_hat, err_var):

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]
        # breakpoint()
        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        outer_dim_size = self._outer_dim_size
        inner_dim_size = self._inner_dim_size

        #####################################
        # Compute the interpolation matrix
        #####################################

        # Computation of the interpolation matrix is done solving the
        # least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        #
        # Computing A
        #

        # Covariance matrices restricted to pilot locations
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = self._pil_cov_mat

        # Adding batch, receive, and receive antennas dimensions to the
        # covariance matrices restricted to pilot locations and to the
        # regularization values
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = expand_to_rank(pil_cov_mat, 8, 0)
        pil_cov_mat = tf.tile(pil_cov_mat, [batch_size, num_rx, num_rx_ant,
                                                     1, 1, 1, 1, 1])

        # Adding the noise variance to the covariance matrices restricted to
        # pilots
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat_ = tf.transpose(pil_cov_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        err_var_ = tf.complex(err_var, self._rzero)
        err_var_ = tf.transpose(err_var_, [3, 4, 5, 6, 0, 1, 2])
        a_mat = tf.tensor_scatter_nd_add(pil_cov_mat_,
                                        self._add_err_var_indices, err_var_)
        a_mat = tf.transpose(a_mat, [5, 6, 7, 0, 1, 2, 3, 4])

        #
        # Computing B
        #

        # B is pre-computed as it only depend on the channel covariance and
        # pilot pattern.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = self._b_mat
        b_mat = expand_to_rank(b_mat, 8, 0)
        b_mat = tf.tile(b_mat, [batch_size, num_rx, num_rx_ant,
                                1, 1, 1, 1, 1])

        #
        # Computing the interpolation matrix
        #

        # Using lstsq to compute the columns of the interpolation matrix
        # corresponding to pilots.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, max_num_pil]
        ext_mat = tf.linalg.lstsq(a_mat, b_mat, fast=False)
        ext_mat = tf.transpose(ext_mat, [0,1,2,3,4,5,7,6], conjugate=True)

        # Filling with zeros the columns not corresponding to pilots.
        # An extra dummy outer dim is added to scatter there the coefficients
        # of the identity matrix used for padding.
        # This dummy dim is then removed.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, inner_dim_size]
        ext_mat = tf.transpose(ext_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        ext_mat = tf.scatter_nd(self._pil_loc, ext_mat,
                                            [num_tx, num_tx_stream,
                                             outer_dim_size,
                                             inner_dim_size+1,
                                             inner_dim_size+1,
                                             batch_size, num_rx, num_rx_ant])
        ext_mat = tf.transpose(ext_mat, [5, 6, 7, 0, 1, 2, 3, 4])
        ext_mat = ext_mat[...,:inner_dim_size,:inner_dim_size]

        ################################################
        # Apply interpolation over the inner dimension
        ################################################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        # breakpoint()
        h_hat = tf.expand_dims(h_hat, axis=-1)
        h_hat = tf.matmul(ext_mat, h_hat)
        h_hat = tf.squeeze(h_hat, axis=-1)

        ##############################
        # Compute the error variances
        ##############################

        # Keep track of the previous estimation error variances for later use
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        err_var_old = err_var

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        cov_mat = expand_to_rank(self._cov_mat, 8, 0)
        err_var = tf.linalg.diag_part(cov_mat)
        err_var_mat = expand_to_rank(self._err_var_mat, 8, 0)
        err_var_mat = tf.transpose(err_var_mat, [0, 1, 2, 3, 4, 5, 7, 6])
        err_var = err_var - tf.reduce_sum(ext_mat*err_var_mat, axis=-1)
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, self._rzero)

        #####################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        #####################################
        if not self._last_step:
            #
            # Variance of h_hat
            #
            # Conjugate transpose of LMMSE matrix
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            ext_mat_h = tf.transpose(ext_mat, [0, 1, 2, 3, 4, 5, 7, 6],
                                     conjugate=True)
            # First part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            h_hat_var_1 = tf.matmul(cov_mat, ext_mat_h)
            h_hat_var_1 = tf.transpose(h_hat_var_1, [0, 1, 2, 3, 4, 5, 7, 6])
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var_1 = tf.reduce_sum(ext_mat*h_hat_var_1, axis=-1)
            # Second part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_old_c = tf.complex(err_var_old, self._rzero)
            err_var_old_c = tf.expand_dims(err_var_old_c, axis=-1)
            h_hat_var_2 = err_var_old_c*ext_mat_h
            h_hat_var_2 = tf.transpose(h_hat_var_2, [0, 1, 2, 3, 4, 5, 7, 6])
            h_hat_var_2 = tf.reduce_sum(ext_mat*h_hat_var_2, axis=-1)
            # Variance of h_hat
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var = h_hat_var_1 + h_hat_var_2
            # Scaling factor
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_c = tf.complex(err_var, self._rzero)
            h_var = tf.linalg.diag_part(cov_mat)
            s = tf.math.divide_no_nan(2.*h_var, h_hat_var + h_var - err_var_c)
            # Apply scaling to estimate
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat = s*h_hat
            # Updated variance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var = s*(s-1.)*h_hat_var + (1.-s)*h_var + s*err_var_c
            err_var = tf.math.real(err_var)
            err_var = tf.maximum(err_var, self._rzero)

        return h_hat, err_var
    

class IndustrialLMMSEInterpolator(BaseChannelInterpolator):
    """
    5G 工业级 LMMSE 信道插值器（Sionna适配）
    集成：PDP提取、CP噪声估计、稀疏多径、Rf自相关、Toeplitz矩阵、调制系数β、分区域插值、多符号合并
    完全对标商用基站LMMSE算法
    """
    def __init__(self, pilot_pattern, fft_size, cp_length, modulation_type, pilot_sym_idx,
                  scs,num_ofdm_symbols, precision=None):
        super().__init__()
        # 精度配置
        if precision is None:
            from sionna.phy.config import Config
            self._precision = Config.precision
        else:
            self._precision = precision
        self._cdtype = tf.complex64 if self._precision == "single" else tf.complex128
        self._rdtype = tf.float32 if self._precision == "single" else tf.float64

        # 系统参数
        self._pilot_pattern = pilot_pattern
        self._fft_size = fft_size          # FFT/IFFT长度
        self._cp_length = cp_length        # OFDM循环前缀长度
        self._modulation_type = modulation_type  # 调制方式（QPSK/16QAM/64QAM）
        # self._max_paths = max_paths        # 稀疏多径最大保留条数
        # self._merge_symbols = merge_symbols# 多符号功率合并开关
        self._beta = self._get_modulation_beta() # 调制系数β
        

        # 解析导频图案
        self._mask = np.array(pilot_pattern.mask)
        self._num_tx, self._num_streams, self._num_syms, self._num_sc = self._mask.shape
        self._pilot_indices = self._get_pilot_indices_per_symbol()
        self._pilot_sym_idx = pilot_sym_idx
        self._scs = scs
        self._num_ofdm_symbols = num_ofdm_symbols
        self._num_rx = 1#修改修改
        self._interpol = LinearInterpolator(self._pilot_pattern)
    def _get_modulation_beta(self):
        """3GPP标准调制系数，修正噪声正则项"""
        if self._modulation_type == 2:
            return tf.constant(1.0, dtype=self._rdtype)
        elif self._modulation_type == 4:
            return tf.constant(17/9, dtype=self._rdtype)
        elif self._modulation_type == 6:
            return tf.constant(2.6857, dtype=self._rdtype)
        else:
            return tf.constant(1.0, dtype=self._rdtype)
    def _get_pilot_indices_per_symbol(self):
        """按符号/天线/流提取导频子载波索引"""
        pilot_indices = []
        for tx in range(self._num_tx):
            for stream in range(self._num_streams):
                sym_indices = []
                for sym in range(self._num_syms):
                    sc_idx = np.where(self._mask[tx, stream, sym])[0]
                    sym_indices.append(sc_idx)
                pilot_indices.append(sym_indices)
        return pilot_indices
    def _DoChannelEstimationPower(self,H_ls_slot, ScsNumPerSymbol, DMRS_Index, nb_tfu):
        # 起始索引，MATLAB从1开始，Python从0开始
        StartInd = 0
        batch = H_ls_slot.shape[0]
        nb_ttu = nb_tfu
        ifft_flag = 0
        if ifft_flag == 1:
            # 计算向上取2的幂次
            val = 6 * nb_ttu
            len_ifft = 2 ** tf.cast(tf.math.ceil(tf.math.log2(tf.cast(val, tf.float32))), tf.int32)
            if len_ifft == 32:
                len_ifft = 64
        else:
            len_ifft = ScsNumPerSymbol[0]

        num_dmrs = len(DMRS_Index)
        pilot_channel_est_power = np.zeros((batch,len_ifft, num_dmrs))
        # breakpoint()
        for i in range(num_dmrs):
            scs_num = ScsNumPerSymbol[i]
            # 截取当前符号导频
            end_ind = StartInd + scs_num
            # breakpoint()
            H_LS_PerSymbol = H_ls_slot[:,StartInd:end_ind]

            if ifft_flag == 1:
                pad_width = [[0, 0], [0, len_ifft - tf.shape(H_LS_PerSymbol)[1]]]
                H_padded = tf.pad(H_LS_PerSymbol, paddings=pad_width)
                H_time = tf.signal.ifft(H_padded)
                scale = tf.sqrt(tf.cast(len_ifft, dtype=H_time.dtype))
                breakpoint()
                H_temp_time_domain = H_time * scale
                # H_temp_time_domain = tf.signal.ifft(H_LS_PerSymbol,len_ifft) * tf.sqrt(tf.cast(len_ifft))
            else:
                # breakpoint()
                current_len = tf.shape(H_LS_PerSymbol)[-1]
                pad_len = len_ifft - current_len
                H_adjusted = tf.pad(H_LS_PerSymbol, paddings=[[0, 0], [0, pad_len]])
                H_temp_time_domain = tf.signal.ifft(H_adjusted)
            chest_time = H_temp_time_domain
            # breakpoint()
            chest_time_abs2_tmp = tf.abs(H_temp_time_domain) ** 2
            pilot_channel_est_power[:,:,i]= chest_time_abs2_tmp
            StartInd = end_ind

        # 加权系数
        scs_tensor = tf.convert_to_tensor(ScsNumPerSymbol, dtype=tf.float32)
        weight_factor = (scs_tensor ** 2) / tf.reduce_sum(scs_tensor ** 2)
        # repmat等价：[num_dmrs,] -> [len_ifft, num_dmrs]
        # weight_factor_rep = tf.tile(tf.expand_dims(weight_factor, axis=0), [len_ifft, 1])
        weight_factor_rep = tf.tile(weight_factor[None, None, :], [batch, len_ifft, 1])
        pilot_channel_est_power = tf.convert_to_tensor(pilot_channel_est_power,dtype=tf.float32)
        # 加权求和
        chest_time_abs2_combine = tf.reduce_sum(pilot_channel_est_power * weight_factor_rep, axis=2)

        return chest_time_abs2_combine

    def _R_f_estimation_V1(self,H_LS_PerSymbol, chest_time_abs2_combine, N0):
        # breakpoint()
        batch_size = tf.shape(H_LS_PerSymbol)[0]
        # combine_flag = myconfigs.DE.fre_che_est_rs_combine.upper() == 'YES'

        # ========== step1: 选择PDP与时域长度 ==========
        # 多列导频平均模式
        FFT_Len_ChanDelay = tf.shape(chest_time_abs2_combine)[1]
        P_PDP = chest_time_abs2_combine
        # 统一计算精度
        P_PDP = tf.cast(P_PDP, tf.float32)
        # FFT_Len_int = FFT_Len_ChanDelay
        FFT_Len_f = tf.cast(FFT_Len_ChanDelay, tf.float32)
        # FFT_Len_f = FFT_Len_ChanDelay

        # ========== step2: 噪声估计与底噪扣除 ==========
        noise_seg_len = FFT_Len_ChanDelay // 4
        start_idx1 = FFT_Len_ChanDelay // 8
        start_idx2 = (5 * FFT_Len_ChanDelay) // 8

        noise_th1 = tf.reduce_sum(P_PDP[:, start_idx1 : start_idx1 + noise_seg_len], axis=-1)
        noise_th2 = tf.reduce_sum(P_PDP[:, start_idx2 : start_idx2 + noise_seg_len], axis=-1)
        # breakpoint()
        noise_power = tf.minimum(noise_th1, noise_th2) * 4.0#
        Gama_fixed = noise_power / tf.cast(noise_seg_len, tf.float32)

        # 去噪，负值置零
        # h_f_abs_fixed = tf.nn.relu(P_PDP - Gama_fixed[:, None])
        diff = P_PDP - Gama_fixed[:, None]
        h_f_abs_fixed = tf.maximum(diff, 0.0)
        # ========== step3: 提取48条多径（16主径+各左右邻径） ==========
        Max_Path_Num = 16
        total_taps = Max_Path_Num * 3  # 共48径
        h_f_abs_max_fixed = tf.zeros([batch_size, total_taps], dtype=tf.float32)
        h_f_abs_max_idx_fixed = tf.zeros([batch_size, total_taps], dtype=tf.int32)
        h_f_abs_current = h_f_abs_fixed  # 迭代用，已选径置零
        batch_idx = tf.range(batch_size)  # 批量更新索引
        
        # 第一条主径手动提取
        max_vals, max_indices = tf.math.top_k(h_f_abs_current, k=1)
        max_vals = tf.squeeze(max_vals, axis=-1)
        max_indices = tf.squeeze(max_indices, axis=-1)

        # 循环边界的左右邻径索引
        left_idx = (max_indices - 1 + FFT_Len_ChanDelay) % FFT_Len_ChanDelay
        right_idx = (max_indices + 1) % FFT_Len_ChanDelay

        # 批量取值
        left_vals = tf.gather(h_f_abs_current, left_idx, batch_dims=1)
        right_vals = tf.gather(h_f_abs_current, right_idx, batch_dims=1)

        # 存入前3个位置
        for pos_offset, vals, idxs in zip([0, 1, 2], [max_vals, left_vals, right_vals], [max_indices, left_idx, right_idx]):
            pos_tensor = tf.fill([batch_size], pos_offset)
            update_indices = tf.stack([batch_idx, pos_tensor], axis=1)
            h_f_abs_max_fixed = tf.tensor_scatter_nd_update(h_f_abs_max_fixed, update_indices, vals)
            h_f_abs_max_idx_fixed = tf.tensor_scatter_nd_update(h_f_abs_max_idx_fixed, update_indices, idxs)

        # 已选径置零
        mask = tf.one_hot(max_indices, depth=FFT_Len_ChanDelay) + \
            tf.one_hot(left_idx, depth=FFT_Len_ChanDelay) + \
            tf.one_hot(right_idx, depth=FFT_Len_ChanDelay)
        h_f_abs_current = h_f_abs_current * (1.0 - mask)

        # 剩余15条主径循环提取
        for k in range(1, Max_Path_Num):
            pos_start = 3 * k

            max_vals, max_indices = tf.math.top_k(h_f_abs_current, k=1)
            max_vals = tf.squeeze(max_vals, axis=-1)
            max_indices = tf.squeeze(max_indices, axis=-1)

            left_idx = (max_indices - 1 + FFT_Len_ChanDelay) % FFT_Len_ChanDelay
            right_idx = (max_indices + 1) % FFT_Len_ChanDelay

            left_vals = tf.gather(h_f_abs_current, left_idx, batch_dims=1)
            right_vals = tf.gather(h_f_abs_current, right_idx, batch_dims=1)

            # 存入对应位置
            for pos_offset, vals, idxs in zip([0, 1, 2], [max_vals, left_vals, right_vals], [max_indices, left_idx, right_idx]):
                pos = pos_start + pos_offset
                pos_tensor = tf.fill([batch_size], pos)
                update_indices = tf.stack([batch_idx, pos_tensor], axis=1)
                h_f_abs_max_fixed = tf.tensor_scatter_nd_update(h_f_abs_max_fixed, update_indices, vals)
                h_f_abs_max_idx_fixed = tf.tensor_scatter_nd_update(h_f_abs_max_idx_fixed, update_indices, idxs)

            # 已选径置零
            mask = tf.one_hot(max_indices, depth=FFT_Len_ChanDelay) + \
                tf.one_hot(left_idx, depth=FFT_Len_ChanDelay) + \
                tf.one_hot(right_idx, depth=FFT_Len_ChanDelay)
            h_f_abs_current = h_f_abs_current * (1.0 - mask)

        # ========== step4: 信号功率与SNR计算 ==========
        signal_power = tf.reduce_sum(h_f_abs_fixed, axis=-1)
        # 保护逻辑：信号功率为0时用前3径功率和代替
        sum_3taps = h_f_abs_max_fixed[:, 0] + h_f_abs_max_fixed[:, 1] + h_f_abs_max_fixed[:, 2]
        signal_power = tf.where(tf.equal(signal_power, 0.0), sum_3taps, signal_power)
        # breakpoint()
        SNR = signal_power / noise_power
        
        SNR_clipped = tf.maximum(SNR, 1e-12)
        SNR_Recip = 1.0 / SNR_clipped
        # has_inf = tf.reduce_any(tf.math.is_inf(SNR_Recip))
        # if has_inf:
        #     breakpoint()
        #     a=0
        # ========== step5: 门限保护 ==========
        PDP_Th_fixed = tf.maximum(Gama_fixed * 4.0, h_f_abs_max_fixed[:, 0] / 128.0)
        h_f_abs_max_fixed = tf.where(h_f_abs_max_fixed < PDP_Th_fixed[:, None], 0.0, h_f_abs_max_fixed)
        # ========== step6: 子载波间隔与时延参数计算 ==========
        sub_carrier_spacing = self._scs
        # tao_cp_ref = myconfigs.sys_para.ncp_length1
        # N_FFT = myconfigs.sys_para.n_fft
        tao_cp_ref = self._cp_length
        log2_val = tf.math.log(tf.cast(self._fft_size, tf.float32)) / tf.math.log(2.0)
        N_FFT = tf.pow(2,tf.math.ceil(log2_val))

        Ts_ref = 1.0 / (2048 * 15e3)
        Ts_current = 1.0 / (N_FFT * sub_carrier_spacing)
        Ts_ration = Ts_ref / Ts_current
        # breakpoint()
        N0_f = tf.cast(N0, tf.float32)
        # N0_f = N0
        tao_A = tf.floor(0.0 * Ts_ration * N0_f * FFT_Len_f) / tf.cast(N_FFT, tf.float32)
        tao_Cp = tao_cp_ref * N0_f * FFT_Len_f / tf.cast(N_FFT, tf.float32)
        tao_leak = tf.floor(100.0 * Ts_ration * N0_f * FFT_Len_f) / tf.cast(N_FFT, tf.float32)

        # ========== step7: 时域索引调整 ==========
        first_idx = tf.cast(h_f_abs_max_idx_fixed[:, 0], tf.float32)
        threshold = tao_A + tao_Cp
        IdxTemp = tf.where(first_idx > threshold, first_idx - FFT_Len_f, first_idx)

        N_Seg = tf.where(IdxTemp - tao_leak < 0, IdxTemp - tao_leak + FFT_Len_f, FFT_Len_f)
        N_FFT2 = N0 * FFT_Len_f
        index_offset = N_FFT2 - FFT_Len_f

        FFT_Index = tf.cast(h_f_abs_max_idx_fixed, tf.float32)
        FFT_Index_adjusted = tf.where(FFT_Index > N_Seg[:, None], FFT_Index + index_offset, FFT_Index)

        # 取前40径做DFT
        FFT_Index_40 = FFT_Index_adjusted[:, :40]
        h_vals_40 = h_f_abs_max_fixed[:, :40]

        # ========== step8: 75点DFT计算频域自相关 ==========
        k_vec = tf.range(75, dtype=tf.float32)
        # 向量化计算所有相位 [batch, 75, 40]
        phi = 2.0 * np.pi * k_vec[None, :, None] * FFT_Index_40[:, None, :] / N_FFT2

        cos_phi = tf.cos(phi)
        sin_phi = -tf.sin(phi)

        # 批量矩阵乘法加权求和
        h_vals_exp = h_vals_40[:, :, None]
        real_dft_fac = tf.squeeze(tf.matmul(cos_phi, h_vals_exp), axis=-1)
        imag_dft_fac = tf.squeeze(tf.matmul(sin_phi, h_vals_exp), axis=-1)
        # breakpoint()
        # 直流分量归一化
        dc_real = real_dft_fac[:, 0:1]
        dc_real_safe = tf.where(tf.equal(dc_real, 0.0), 1.0, dc_real)
        real_dft_fac = real_dft_fac / dc_real_safe
        imag_dft_fac = imag_dft_fac / dc_real_safe

        # 强制直流分量为1+0j
        dc_pos = tf.stack([batch_idx, tf.zeros_like(batch_idx)], axis=1)
        real_dft_fac = tf.tensor_scatter_nd_update(real_dft_fac, dc_pos, tf.ones(batch_size, dtype=tf.float32))
        imag_dft_fac = tf.tensor_scatter_nd_update(imag_dft_fac, dc_pos, tf.zeros(batch_size, dtype=tf.float32))

        R_f = tf.complex(real_dft_fac, imag_dft_fac)

        return R_f, SNR_Recip, signal_power, noise_power
    
    def _LMMSE_chan_est_block_method(self,H_LS, R_f, SNR_Recip, N_DL_RB, NO, modulation_mode):

        coeff = 2
        rs_num = int(coeff * 12 / NO)
        batch_size = tf.shape(H_LS)[0]

        # ========== 1. 计算beta系数 ==========
        if modulation_mode == 'QPSK':
            beta = 1.0
        elif modulation_mode == '16QAM':
            beta = 17.0 / 9.0
        elif modulation_mode == '64QAM':
            beta = 2.6857
        else:
            beta = 1.0
        
        # ========== 2. 生成自相关矩阵 FacMatrix ==========
        # 生成索引矩阵（0基）
        i_vec = tf.range(rs_num, dtype=tf.int32)
        j_vec = tf.range(rs_num, dtype=tf.int32)
        i_mat = i_vec[:, None]  # [rs_num, 1]
        j_mat = j_vec[None, :]  # [1, rs_num]
        diff = tf.abs(i_mat - j_mat)
        delay_idx = diff * NO  # 延迟子载波数，对应R_f的0基索引

        # 从自相关序列中取值，上三角取共轭
        R_f_vals = tf.gather(R_f, delay_idx, axis=1)  # [batch, rs_num, rs_num]
        FacMatrix = tf.where(i_mat < j_mat, tf.math.conj(R_f_vals), R_f_vals)

        # 叠加噪声对角项
        diag_vals = beta * SNR_Recip  # [batch]
        diag_vals_exp = tf.tile(diag_vals[:, None], [1, rs_num])
        diag_mat = tf.linalg.diag(diag_vals_exp)
        diag_mat = tf.cast(diag_mat, dtype=FacMatrix.dtype)
        FacMatrix = FacMatrix + diag_mat
        
        # ========== 3. 生成三段互相关矩阵 ==========
        # 3.1 开头部分 FccMatrix_part1
        mSize1 = int(rs_num * NO / 2)
        i_p1 = tf.range(mSize1, dtype=tf.int32)[:, None]
        j_p1 = tf.range(rs_num, dtype=tf.int32)[None, :]
        index_p1 = NO * j_p1 - i_p1
        delay_p1 = tf.abs(index_p1)

        R_f_p1 = tf.gather(R_f, delay_p1, axis=1)
        FccMatrix_part1 = tf.where(index_p1 > 0, tf.math.conj(R_f_p1), R_f_p1)
        # 3.2 中间部分 FccMatrix_part2
        mSize2 = NO
        i_p2 = tf.range(mSize2, dtype=tf.int32)[:, None]
        j_p2 = tf.range(rs_num, dtype=tf.int32)[None, :]
        term_p2 = (tf.cast(j_p2, tf.float32) - rs_num/2 + 1.0) * NO
        index_p2 = tf.cast(i_p2, tf.float32) - term_p2
        delay_p2 = tf.cast(tf.abs(index_p2), tf.int32)

        R_f_p2 = tf.gather(R_f, delay_p2, axis=1)
        FccMatrix_part2 = tf.where(index_p2 < 0, tf.math.conj(R_f_p2), R_f_p2)

        # 3.3 结尾部分 FccMatrix_part3
        mSize3 = mSize1
        i_p3 = tf.range(mSize3, dtype=tf.int32)[:, None]
        j_p3 = tf.range(rs_num, dtype=tf.int32)[None, :]
        term_p3 = (tf.cast(j_p3, tf.float32) - rs_num/2) * NO
        index_p3 = tf.cast(i_p3, tf.float32) - term_p3
        delay_p3 = tf.cast(tf.abs(index_p3), tf.int32)

        R_f_p3 = tf.gather(R_f, delay_p3, axis=1)
        FccMatrix_part3 = tf.where(index_p3 < 0, tf.math.conj(R_f_p3), R_f_p3)

        # ========== 4. 求逆得到插值系数矩阵 ==========
        # breakpoint()
        # has_nan = tf.reduce_any(tf.math.is_nan(tf.math.real(FacMatrix)))
        # has_inf = tf.reduce_any(tf.math.is_inf(tf.math.real(FacMatrix)))
        # if has_nan or has_inf:
        #     a=0
        #     breakpoint()
        # eps = 1e-3
        # batch_size = tf.shape(FacMatrix)[0]  # 动态获取batch维度大小
        # # 直接生成形状为 [batch, 12, 12] 的批量单位矩阵
        # eye_mat = tf.eye(tf.shape(FacMatrix)[-1], batch_shape=[batch_size], dtype=FacMatrix.dtype)
        # # 同形状直接相加，每个样本对角线上都加上eps
        # FacMatrix = FacMatrix + eps * eye_mat
        FacMatrix_inv = tf.linalg.inv(FacMatrix)
        FocGen1 = tf.matmul(FccMatrix_part1, FacMatrix_inv)  # [batch, mSize1, rs_num]
        FocGen2 = tf.matmul(FccMatrix_part2, FacMatrix_inv)  # [batch, mSize2, rs_num]
        FocGen3 = tf.matmul(FccMatrix_part3, FacMatrix_inv)  # [batch, mSize3, rs_num]
        # breakpoint()#debug到这儿了
        # ========== 5. 分区域计算信道估计 ==========
        band_num = N_DL_RB * 12

        # 5.1 开头区域：前rs_num个导频 + FocGen1
        H_LS_head = H_LS[:, :rs_num]
        # breakpoint()
        che_est_head = tf.reduce_sum(FocGen1 * H_LS_head[:, None, :], axis=-1)

        # 5.2 结尾区域：后rs_num个导频 + FocGen3
        H_LS_tail = H_LS[:, -rs_num:]
        che_est_tail = tf.reduce_sum(FocGen3 * H_LS_tail[:, None, :], axis=-1)

        # 5.3 中间区域：滑动导频窗口 + FocGen2
        mid_len = band_num - 2 * mSize1
        num_windows = mid_len // 2  # 对齐原MATLAB mod(i,2) 硬编码逻辑

        # 生成每个窗口的导频起始索引（0基，初始值对应原MATLAB的2）
        pilot_starts = 1 + tf.range(num_windows, dtype=tf.int32)
        # 生成所有窗口的导频索引矩阵
        pilot_indices = pilot_starts[:, None] + tf.range(rs_num, dtype=tf.int32)[None, :]
        # 批量取出所有窗口的导频
        H_ls_windows = tf.gather(H_LS, pilot_indices, axis=1)  # [batch, num_windows, rs_num]

        # 批量计算每个窗口的子载波估计
        che_est_mid_win = tf.reduce_sum(
            H_ls_windows[:, :, None, :] * FocGen2[:, None, :, :],
            axis=-1
        )  # [batch, num_windows, mSize2]
        che_est_mid = tf.reshape(che_est_mid_win, [batch_size, mid_len])

        # 拼接三段结果
        che_est = tf.concat([che_est_head, che_est_mid, che_est_tail], axis=1)

        return che_est
    def _DoChannelEstimationOfLMMSEMethod_V1(self,h_hat, tmp, scsnumpersym):
        [batch,num_rx,Rx_ant_num,num_tx,TX_stream,numpilot_per_slot] = h_hat.shape
        
        NO = 2 #pilot comb
        nb_tfu = int(self._fft_size/12)

        slot_pilot_channel_est = np.zeros((batch,Rx_ant_num,TX_stream,self._fft_size,len(self._pilot_sym_idx)),dtype=np.complex64)
        modulation_mode = self._modulation_type
        
        pilot_indices_per_slot = self._pilot_indices[0]
        # scsnumpersym = [pilot_indices_per_slot[idx] for idx in self._pilot_sym_idx]
        noise_power_ante = np.zeros((batch,TX_stream,Rx_ant_num))
        for n_rxva in range(Rx_ant_num):
            noise_power_ofdm_sym = np.zeros((batch,TX_stream,len(self._pilot_sym_idx)))
            for n_txva in range(TX_stream):
                StartInd = 0
                h_ls_slot = tf.squeeze(h_hat[:,:,n_rxva,:,n_txva,:])#一个slot上所有的Hls
                # breakpoint()
                #%多列导频求平均功率
                chest_time_abs2_combine = self._DoChannelEstimationPower(h_ls_slot,scsnumpersym,self._pilot_sym_idx,nb_tfu)
                for j in range(len(self._pilot_sym_idx)):
                    scs_num = scsnumpersym[j]
                    end_ind = StartInd+scs_num
                    H_LS_PerSymbol = h_ls_slot[:,StartInd:end_ind]
                    N_RB = nb_tfu
                    # breakpoint()
                    R_f, SNR_Recip, signal_power, noise_power = self._R_f_estimation_V1(H_LS_PerSymbol, chest_time_abs2_combine, NO)
                    # print('rx_nate,tx_ante,rx_nate:',n_rxva,n_txva,j)
                    # breakpoint()
                    channel_est_per_symbol = self._LMMSE_chan_est_block_method(H_LS_PerSymbol, R_f, SNR_Recip, N_RB, NO, modulation_mode)
                    if channel_est_per_symbol.shape[1] == self._fft_size:
                        slot_pilot_channel_est[:,n_rxva, n_txva, :, j] = channel_est_per_symbol
                    noise_power_ofdm_sym[:,n_txva,j] = noise_power
                    StartInd = end_ind
                weight_factor = scsnumpersym / tf.reduce_sum(scsnumpersym)
                weight_factor_batch = tf.tile(weight_factor[tf.newaxis, :], [batch, 1])
                noise_power_ante[:,n_txva,n_rxva] = tf.reduce_sum(noise_power_ofdm_sym[:,n_txva,:]*weight_factor_batch,axis = -1)
        noise_power_slot = tf.reduce_mean(noise_power_ante,axis=-1)  
        return slot_pilot_channel_est,noise_power,noise_power_ante
    

    def _linear_interp(self,pilot_pos, pilot_reshaped, target_pos):
        a=0
    def _DoInterpProcess(self,symbol_index, chan_time_index):
        # breakpoint()
        symbol_num = tf.shape(symbol_index)[0]
        chan_len = tf.shape(chan_time_index)[0]

        CoeffMatrix = tf.zeros([symbol_num, chan_len], dtype=tf.float32)
        symbol_num = tf.shape(symbol_index)[0]  # 包含导频符号
        coeff = tf.ones([symbol_num, 2], dtype=tf.float32)
        coeff = tf.concat([-1 * coeff[:, 0:1], coeff[:, 1:2]], axis=1)

        dmrs_dist = tf.cast(chan_time_index[1] - chan_time_index[0], tf.float32)

        delta = tf.cast(symbol_index, tf.float32) - tf.cast(chan_time_index[0], tf.float32)
        delta_col = delta[:, tf.newaxis]  # 对应MATLAB中转置为列向量的操作
        coeff = delta_col * coeff

        coeff = tf.concat([coeff[:, 0:1] + dmrs_dist, coeff[:, 1:2]], axis=1)
        coeff = coeff / dmrs_dist

        CoeffMatrix = coeff

        return CoeffMatrix
    def _ToDoComputeInterpCoeff(self,symbol_index, chan_time_index):
        # breakpoint()
        len_chan = tf.shape(chan_time_index)[0]
        len_sym = len(symbol_index)

        if len_chan == 1:
            # 只有一列导频不插值，生成单位矩阵
            CoeffMatrix = tf.eye(len_sym)
        elif len_chan == 2:
            # 只有两列导频 直接插值
            CoeffMatrix = self._DoInterpProcess(symbol_index, chan_time_index)
        else:
            # 3列以上导频 需要分段计算系数
            CoeffMatrix = tf.zeros([len_sym, len_chan], dtype=tf.float32)

            # 等价 MATLAB: [~, Ia, ~] = intersect(symbol_index, chan_time_index)
            # 输入均为升序排列，返回交集元素在 symbol_index 中的 0 基索引
            Ia = tf.searchsorted(symbol_index, chan_time_index, side='left')
            len_Ia = len(Ia)

            # 对应 MATLAB: for i = 2:length(Ia)
            for i in range(1, len_Ia):
                if i == 1:
                    # 第一段：开头到第2个导频位置
                    symbol_index_temp = symbol_index[:Ia[i]+1]
                    chan_time_index_temp = chan_time_index[:2]
                    coeff_block = self._DoInterpProcess(symbol_index_temp, chan_time_index_temp)

                    # 写入系数矩阵对应区域：行 0~Ia[i]，列 0~1
                    rows = tf.range(Ia[i] + 1)
                    cols = tf.range(2)
                    ii, jj = tf.meshgrid(rows, cols, indexing='ij')
                    indices = tf.stack([tf.reshape(ii, [-1]), tf.reshape(jj, [-1])], axis=1)
                    updates = tf.reshape(coeff_block, [-1])
                    CoeffMatrix = tf.tensor_scatter_nd_update(CoeffMatrix, indices, updates)

                elif i == len_Ia - 1:
                    # 最后一段：倒数第2个导频的下一个位置到结尾
                    symbol_index_temp = symbol_index[Ia[i-1]+1:]
                    chan_time_index_temp = chan_time_index[-2:]
                    coeff_block = self._DoInterpProcess(symbol_index_temp, chan_time_index_temp)

                    # 写入系数矩阵对应区域：行 Ia[i-1]+1 ~ 末尾，列 i-1 ~ i
                    rows = tf.range(Ia[i-1] + 1, len_sym)
                    cols = tf.range(i-1, i+1)
                    ii, jj = tf.meshgrid(rows, cols, indexing='ij')
                    indices = tf.stack([tf.reshape(ii, [-1]), tf.reshape(jj, [-1])], axis=1)
                    updates = tf.reshape(coeff_block, [-1])
                    CoeffMatrix = tf.tensor_scatter_nd_update(CoeffMatrix, indices, updates)

                else:
                    # 中间段：相邻两个导频之间的区域
                    symbol_index_temp = symbol_index[Ia[i-1]+1 : Ia[i]+1]
                    chan_time_index_temp = chan_time_index[i-1 : i+1]
                    coeff_block = self._DoInterpProcess(symbol_index_temp, chan_time_index_temp)

                    # 写入系数矩阵对应区域：行 Ia[i-1]+1 ~ Ia[i]，列 i-1 ~ i
                    rows = tf.range(Ia[i-1] + 1, Ia[i] + 1)
                    cols = tf.range(i-1, i+1)
                    ii, jj = tf.meshgrid(rows, cols, indexing='ij')
                    indices = tf.stack([tf.reshape(ii, [-1]), tf.reshape(jj, [-1])], axis=1)
                    updates = tf.reshape(coeff_block, [-1])
                    CoeffMatrix = tf.tensor_scatter_nd_update(CoeffMatrix, indices, updates)

        return CoeffMatrix

    def _mmse_time_interpolate(self,pilot_channelest,pilot_sym_idx):
        # breakpoint()
        mp = []
        pilot_sym_num = len(pilot_sym_idx)
        # symbol_index = tf.cast(symbol_index, tf.int32)
        # symbol_index_0 = symbol_index  # 统一转为0基索引
        batch,scs,pilot_sym_ = pilot_channelest.shape
        # ========== Step1: 展平导频信道为二维 [频域点数, 导频符号数] ==========
        # pilot_reshaped = tf.reshape(pilot_channelest, [-1, pilot_sym_num])
        pilot_reshaped = pilot_channelest
        symbol_num = self._num_ofdm_symbols
        chan_time_index = [i for i in range(symbol_num)]

        # ========== Step2: 频域处理 ==========
        criteria = 'non-MSE'
        if 'MMSE' in criteria:
            # MMSE频域滤波：向量化替代循环，结果完全一致
            pilot_expand = tf.transpose(pilot_reshaped)[..., None]  # [导频数, 输入频点, 1]
            chan_freq = tf.squeeze(tf.matmul(mp, pilot_expand), axis=-1)  # [导频数, 输出频点]
            chan_freq = tf.transpose(chan_freq)  # [输出频点, 导频数]
            freq_num = tf.shape(chan_freq)[0]
        elif pilot_sym_num == 1:
            # SRS_LS：频域2倍线性插值，带外推
            freq_in = tf.shape(pilot_reshaped)[0]
            freq_num = freq_in * 2
            pilot_pos = tf.range(0, freq_num, 2, dtype=tf.int32)
            target_pos = tf.range(freq_num, dtype=tf.int32)
            chan_freq_0 = self._linear_interp(pilot_pos, pilot_reshaped[:, 0], target_pos)
            chan_freq = chan_freq_0[:, None]  # [频点数, 1]
        else:
            # DRS_LS：直接使用导频值，不做频域插值
            chan_freq = pilot_reshaped
            freq_num = tf.shape(chan_freq)[1]

        # 初始化输出矩阵
        frequency_channel = tf.zeros([batch,freq_num, symbol_num], dtype=chan_freq.dtype)

        # ========== Step3: 计算有效数据符号范围 ==========
        # unused_sym = range(1,self._num_ofdm_symbols)#tf.cast(myconfigs.sys_para.unused_symbol_position, tf.int32)
        data_sym_idx = chan_time_index 
        SectionNum = 1

        frequency_index = [0,self._num_sc/12]
        TimeDomain_index = [data_sym_idx[0], data_sym_idx[-1]]

        # # ========== Step6: 逐段时域插值 ==========
        for i in range(SectionNum):
            chan_time_index_input = data_sym_idx
            mask_matrix = tf.equal(tf.expand_dims(pilot_sym_idx, axis=1), tf.expand_dims(chan_time_index_input, axis=0))
            mask = tf.reduce_any(mask_matrix, axis=1)  # 每个导频是否落在当前区间内

            # 得到对应索引与筛选后的导频位置
            IA = tf.where(mask)[:, 0]  # 0基索引，对应 MATLAB 的 1 基 IA
            symbol_index_input = tf.gather(pilot_sym_idx, IA)
            CoeffMatrix = self._ToDoComputeInterpCoeff(chan_time_index_input, symbol_index_input)
            # breakpoint()
            if SectionNum != 1:
                base_prb = frequency_index[0, 0]  # 基准PRB，对应MATLAB frequency_index(1,1)
                prb_start = frequency_index[i, 0]
                prb_end = frequency_index[i, 1]
                sc_start = (prb_start - base_prb) * 12    # 起始子载波索引（0基，包含）
                sc_end = (prb_end - base_prb + 1) * 12    # 结束子载波索引（0基，不包含）

                # ========== 批量提取对应子载波的导频信道 ==========
                # chan_freq形状：[batch, 总子载波数, 导频符号数]
                # chan_sub形状：[batch, 当前段子载波数, 导频符号数]
                chan_sub = chan_freq[:,sc_start:sc_end,:]
            else:
                chan_sub = chan_freq
            # ========== 筛选有效导频位置 ==========
            # 对应MATLAB: y = y(IA)
            # IA为0基导频索引，输出形状：[batch, 当前段子载波数, len(IA)]
            y = tf.gather(chan_sub, IA, axis=2)

            # ========== 批量矩阵乘法完成时域插值 ==========
            # 对应MATLAB: temp = CoeffMatrix * y
            # CoeffMatrix形状：[目标符号数, len(IA)]
            # 输出temp形状：[batch, 当前段子载波数, 目标符号数]
            CoeffMatrix = tf.cast(CoeffMatrix, y.dtype)
            temp = tf.matmul(y, CoeffMatrix, transpose_b=True)
            frequency_channel = temp
            # # ========== 批量写入结果到frequency_channel ==========
            # # 对应MATLAB循环内逐行赋值，向量化实现，性能远高于逐子载波循环
            # batch_size = tf.shape(chan_freq)[0]
            # sc_indices = tf.range(sc_start, sc_end)
            # sym_indices = chan_time_index_input

            # # 生成三维网格索引，对应 batch、子载波、符号 三个维度
            # batch_grid, sc_grid, sym_grid = tf.meshgrid(
            #     tf.range(batch_size),
            #     sc_indices,
            #     sym_indices,
            #     indexing='ij'
            # )
            # indices = tf.stack([
            #     tf.reshape(batch_grid, [-1]),
            #     tf.reshape(sc_grid, [-1]),
            #     tf.reshape(sym_grid, [-1])
            # ], axis=1)
            # updates = tf.reshape(temp, [-1])

            # # 执行张量区域更新
            # frequency_channel = tf.tensor_scatter_nd_update(frequency_channel, indices, updates)

        subframe_channel_est = frequency_channel
        # subframe_channel_est = 0
        return subframe_channel_est

        # return time_interpol_perslot

    def _interpolate_zxx(self, h_hat, err_var):
        
        pilot_indices_per_slot = self._pilot_indices[0]
        scsnumpersym = []
        for idx in self._pilot_sym_idx:
            scsnumpersym.append(len(pilot_indices_per_slot[idx]))
        #频域插值
        slot_pilot_channel_est, noise_power, noise_power_ante = self._DoChannelEstimationOfLMMSEMethod_V1(h_hat, self._pilot_sym_idx, scsnumpersym)
        # 时域插值
        batch,RX_NUM,TX_NUM,scs,pilot_sym_num=slot_pilot_channel_est.shape
        subframe_channel_est=np.zeros((batch,self._num_rx,RX_NUM,self._num_tx,TX_NUM,self._num_ofdm_symbols,scs),complex)
        for nrxva in range(RX_NUM):
            for n_txva in range(TX_NUM):
                # breakpoint()
                pilot_channelest = slot_pilot_channel_est[:,nrxva,n_txva,:,:]
                time_interpol_perslot = self._mmse_time_interpolate(pilot_channelest,self._pilot_sym_idx)
                time_interpol_reshaped = tf.transpose(time_interpol_perslot, perm=[0, 2, 1])
                # breakpoint()
                subframe_channel_est[:,0,nrxva,0,n_txva,:,:] = time_interpol_reshaped
        err_var = tf.cast(err_var, tf.complex64)
        err_var = self._interpol._interpolate(err_var)#linear 插值
        err_var = tf.math.real(err_var)
        return subframe_channel_est,err_var

    def __call__(self, h_hat, err_var):
        """Sionna标准接口调用"""
        return self._interpolate_zxx(h_hat, err_var)

    
class SpatialChannelFilter(Object):
    # pylint: disable=line-too-long
    r"""
    Implements linear minimum mean square error (LMMSE) smoothing

    We consider the following model:

    .. math::

        \mathbf{y} = \mathbf{h} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated
    with covariance matrix
    :math:`\mathbb{E}\left[ \mathbf{h} \mathbf{h}^{\mathsf{H}} \right] = \mathbf{R}`,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`.

    The channel estimate :math:`\hat{\mathbf{h}}` is computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{A} \mathbf{y}

    where

    .. math::

        \mathbf{A} = \mathbf{R} \left( \mathbf{R} + N_0 \mathbf{I}_M \right)^{-1}

    where :math:`\mathbf{I}_M` is the :math:`M \times M` identity matrix.
    The estimation error is:

    .. math::

        \tilde{h} = \mathbf{h} - \hat{\mathbf{h}}

    The error variances

    .. math::

             \sigma^2_i = \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right], 0 \leq i \leq M-1

    are the diagonal elements of

    .. math::

        \mathbb{E}\left[\mathbf{\tilde{h}} \mathbf{\tilde{h}}^{\mathsf{H}} \right] = \mathbf{R} - \mathbf{A}\mathbf{R}.

    Parameters
    ----------
    cov_mat : [num_rx_ant, num_rx_ant], `tf.complex`
        Spatial covariance matrix of the channel

    last_step : `bool`
        Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    Input
    -----
    h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `tf.complex`
        Channel estimates

    err_var : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `tf.float`
        Channel estimation error variances

    Output
    ------
    h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `tf.complex`
        Channel estimates smoothed accross the spatial dimension

    err_var : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `tf.float`
        The channel estimation error variances of the smoothed channel estimates
    """
    def __init__(self, cov_mat, last_step):
        if cov_mat.dtype==tf.complex64:
            precision = "single"
        elif cov_mat.dtype==tf.complex128:
            precision = "double"
        else:
            msg = "`cov_mat` dtype must be one of tf.complex64 or tf.complex128"
            raise TypeError(msg)
        super().__init__(precision=precision)
        self._rzero = tf.zeros((), self.rdtype)
        self._cov_mat = cov_mat
        self._last_step = last_step

        # Indices for adding a tensor of vectors [..., num_rx_ant] to the
        # diagonal of a tensor of matrices [..., num_rx_ant, num_rx_ant]
        num_rx_ant = cov_mat.shape[0]
        add_diag_indices = [[rxa, rxa] for rxa in range(num_rx_ant)]
        self._add_diag_indices = tf.cast(add_diag_indices, tf.int32)

    def __call__(self, h_hat, err_var):
        # h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_subcarriers, num_rx_ant]
        # err_var : [batch_size, num_rx, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_subcarriers, num_rx_ant]

        # [..., num_rx_ant]
        err_var = tf.complex(err_var, self._rzero)
        # Keep track of the previous estimation error variances for later use
        err_var_old = err_var

        # [num_rx_ant, num_rx_ant]
        cov_mat = self._cov_mat
        cov_mat_t = tf.transpose(cov_mat)
        num_rx_ant = tf.shape(cov_mat)[0]

        ##########################################
        # Compute LMMSE matrix
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat = expand_to_rank(cov_mat, tf.rank(err_var)+1, axis=0)

        # Adding the error variances to the diagonal
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = tf.broadcast_to(cov_mat, tf.concat([tf.shape(err_var),
                                                        [num_rx_ant]], axis=0))
        # [num_rx_ant, ...]
        err_var_ = tf.transpose(err_var, [6, 0, 1, 2, 3, 4, 5])
        # [num_rx_ant, num_rx_ant, ...]
        lmmse_mat = tf.transpose(lmmse_mat, [6, 7, 0, 1, 2, 3, 4, 5])
        lmmse_mat = tf.tensor_scatter_nd_add(lmmse_mat,
                                            self._add_diag_indices, err_var_)
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = tf.transpose(lmmse_mat, [2, 3, 4, 5, 6, 7, 0, 1])

        # [..., num_rx_ant, num_rx_ant]
        l = tf.linalg.cholesky(lmmse_mat)
        lmmse_mat = tf.linalg.cholesky_solve(l, cov_mat)
        lmmse_mat = tf.linalg.adjoint(lmmse_mat)

        ##########################################
        # Apply smoothing
        ##########################################

        # [..., num_rx_ant, 1]
        h_hat = tf.expand_dims(h_hat, axis=-1)
        # [..., num_rx_ant]
        h_hat = tf.squeeze(tf.matmul(lmmse_mat, h_hat), axis=-1)

        ##########################################
        # Compute the estimation error variances
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_t = expand_to_rank(cov_mat_t, tf.rank(lmmse_mat), axis=0)
        # [..., num_rx_ant]
        err_var = tf.reduce_sum(cov_mat_t*lmmse_mat, axis=-1)
        # [..., num_rx_ant]
        err_var = tf.linalg.diag_part(cov_mat) - err_var
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, self._rzero)

        ##########################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        ##########################################
        if not self._last_step:
            #
            # Variance of h_hat
            #
            # Conjugate transpose of the LMMSE matrix
            # [..., num_rx_ant, num_rx_ant]
            lmmse_mat_h = tf.transpose(lmmse_mat, [0, 1, 2, 3, 4, 5, 7, 6],
                                        conjugate=True)
            # First part of the estimate covariance
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_1 = tf.matmul(cov_mat, lmmse_mat_h)
            h_hat_var_1 = tf.transpose(h_hat_var_1, [0, 1, 2, 3, 4, 5, 7, 6])
            # [..., num_rx_ant]
            h_hat_var_1 = tf.reduce_sum(lmmse_mat*h_hat_var_1, axis=-1)
            # Second part of the estimate covariance
            # [..., num_rx_ant, 1]
            err_var_old = tf.expand_dims(err_var_old, axis=-1)
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = err_var_old*lmmse_mat_h
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = tf.transpose(h_hat_var_2, [0, 1, 2, 3, 4, 5, 7, 6])
            # [..., num_rx_ant]
            h_hat_var_2 = tf.reduce_sum(lmmse_mat*h_hat_var_2, axis=-1)
            # Variance of h_hat
            # [..., num_rx_ant]
            h_hat_var = h_hat_var_1 + h_hat_var_2
            # Scaling factor
            # [..., num_rx_ant]
            err_var_c = tf.complex(err_var, self._rzero)
            h_var = tf.linalg.diag_part(cov_mat)
            s = tf.math.divide_no_nan(2.*h_var, h_hat_var + h_var - err_var_c)
            # Apply scaling to estimate
            # [..., num_rx_ant]
            h_hat = s*h_hat
            # Updated variance
            # [..., num_rx_ant]
            err_var = s*(s-1.)*h_hat_var + (1.-s)*h_var + s*err_var_c
            err_var = tf.math.real(err_var)
            err_var = tf.maximum(err_var, self._rzero)

        return h_hat, err_var

class LMMSEInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""
    LMMSE interpolation on a resource grid with optional spatial smoothing

    This class computes for each element of an OFDM resource grid
    a channel estimate and error variance
    through linear minimum mean square error (LMMSE) interpolation/smoothing.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.phy.ofdm.PilotPattern`.

    Depending on the value of ``order``, the interpolation is carried out
    accross time (t), i.e., OFDM symbols, frequency (f), i.e., subcarriers,
    and optionally space (s), i.e., receive antennas, in any desired order.

    For simplicity, we describe the underlying algorithm assuming that interpolation
    across the sub-carriers is performed first, followed by interpolation across
    OFDM symbols, and finally by spatial smoothing across receive
    antennas.
    The algorithm is similar if interpolation and/or smoothing are performed in
    a different order.
    For clarity, antenna indices are omitted when describing frequency and time
    interpolation, as the same process is applied to all the antennas.

    The input ``h_hat`` is first reshaped to a resource grid
    :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`, by scattering the channel
    estimates at pilot locations according to the ``pilot_pattern``. :math:`N`
    denotes the number of OFDM symbols and :math:`M` the number of sub-carriers.

    The first pass consists in interpolating across the sub-carriers:

    .. math::
        \hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` is the OFDM symbol index and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` matrix such that:

    .. math::
        \mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal

    where

    .. math::
        \bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2

    and :math:`\mathbf{R^{(f)}}` is the :math:`M \times M` channel frequency covariance matrix,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads :math:`K_n`
    values to a vector of size :math:`M` according to the ``pilot_pattern`` for the :math:`n^{\text{th}}` OFDM symbol,
    and :math:`\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}` is the channel estimation error covariance built from
    ``err_var`` and assumed to be diagonal.
    Computation of :math:`\bar{\mathbf{A}}_n` is done using an algorithm based on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance matrices.

    The channel estimation error variances after the first interpolation pass are computed as

    .. math::
        \mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size :math:`M \times M` that zeros the
    columns corresponding to sub-carriers not carrying any pilots.
    Note that interpolation is not performed for OFDM symbols which do not carry pilots.

    **Remark**: The interpolation matrix differs across OFDM symbols as different
    OFDM symbols may carry pilots on different sub-carriers and/or have different
    estimation error variances.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next interpolation step, and the error variances are updated accordingly:

    .. math::
        \begin{align}
            \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
            \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
        \end{align}

    where the scaling factor :math:`s_{n,m}` is such that:


    .. math::
        \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}

    which leads to:

    .. math::
        \begin{align}
            s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
        \end{align}

    The second pass consists in interpolating across the OFDM symbols:

    .. math::
        \hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m

    where :math:`1 \leq m \leq M` is the sub-carrier index and :math:`\tilde{\mathbf{h}}^{(2)}_m` is
    the :math:`m^{\text{th}}` column of

    .. math::
        \hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                                    {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                                    \vdots\\
                                    {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                                 \end{bmatrix}

    and :math:`\mathbf{B}_m` is the :math:`N \times N` interpolation LMMSE matrix:

    .. math::
        \mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal

    where

    .. math::
        \bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2

    where :math:`\mathbf{R^{(t)}}` is the :math:`N \times N` channel time covariance matrix,
    :math:`\tilde{\mathbf{\Pi}}_m` the :math:`N \times L_m` matrix that spreads :math:`L_m`
    values to a vector of size :math:`N` according to the ``pilot_pattern`` for the :math:`m^{\text{th}}` sub-carrier,
    and :math:`\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}` is the diagonal matrix of channel estimation error variances
    built by gathering the error variances from (:math:`\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N`) corresponding
    to resource elements carried by the :math:`m^{\text{th}}` sub-carrier.
    Computation of :math:`\bar{\mathbf{B}}_m` is done using an algorithm based on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance matrices.

    The resulting channel estimate for the resource grid is

    .. math::
        \hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]

    The resulting channel estimation error variances are the diagonal coefficients of the matrices

    .. math::
        \mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M

    where :math:`\tilde{\mathbf{\Xi}}_m` is the diagonal matrix of size :math:`N \times N` that zeros the
    columns corresponding to OFDM symbols not carrying any pilots.

    **Remark**: The interpolation matrix differs across sub-carriers as different
    sub-carriers may have different estimation error variances computed by the first
    pass.
    However, all sub-carriers carry at least one channel estimate as a result of
    the first pass, ensuring that a channel estimate is computed for all the resource
    elements after the second pass.

    **Remark:** LMMSE interpolation requires knowledge of the time and frequency
    covariance matrices of the channel. The notebook `OFDM MIMO Channel Estimation and Detection <../tutorials/OFDM_MIMO_Detection.ipynb>`_ shows how to estimate
    such matrices for arbitrary channel models.
    Moreover, the functions :func:`~sionna.phy.ofdm.tdl_time_cov_mat`
    and :func:`~sionna.phy.ofdm.tdl_freq_cov_mat` compute the expected time and frequency
    covariance matrices, respectively, for the :class:`~sionna.phy.channel.tr38901.TDL` channel models.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next smoothing step, and the
    error variances are updated accordingly:

    .. math::
        \begin{align}
            \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
            \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
        \end{align}

    where:

    .. math::
        \begin{align}
            \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
        \end{align}

    Finally, a spatial smoothing step is applied to every resource element carrying
    a channel estimate.
    For clarity, we drop the resource element indexing :math:`(n,m)`.
    We denote by :math:`L` the number of receive antennas, and by
    :math:`\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}` the spatial covariance matrix.

    LMMSE spatial smoothing consists in the following computations:

    .. math::
        \hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}

    where

    .. math::
        \mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.

    The estimation error variances are the digonal coefficients of

    .. math::
        \mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}

    The smoothed channel estimate :math:`\hat{\mathbf{h}}^{(5)}` and corresponding
    error variances :math:`\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)` are
    returned for every resource element :math:`(m,n)`.

    **Remark:** No scaling is performed after the last interpolation or smoothing
    step.

    **Remark:** All passes assume that the estimation error covariance matrix
    (:math:`\mathbf{\Sigma}`, :math:`\tilde{\mathbf{\Sigma}}^{(2)}`, or :math:`\tilde{\mathbf{\Sigma}}^{(4)}`) is diagonal, which
    may not be accurate. When this assumption does not hold, this interpolator is only
    an approximation of LMMSE interpolation.

    **Remark:** The order in which frequency interpolation, temporal
    interpolation, and, optionally, spatial smoothing are applied, is controlled using the
    ``order`` parameter.

    Note
    ----
    This block does not support graph mode with XLA.

    Parameters
    ----------
    pilot_pattern : :class:`~sionna.phy.ofdm.PilotPattern`
        Used pilot pattern

    cov_mat_time : [num_ofdm_symbols, num_ofdm_symbols], `tf.complex`
        Time covariance matrix of the channel

    cov_mat_freq : [fft_size, fft_size], `t`f.complex`
        Frequency covariance matrix of the channel

    cov_time_space : `None` (default) | [num_rx_ant, num_rx_ant], `tf.complex`
        Spatial covariance matrix of the channel.
        Only required if spatial smoothing is requested (see ``order``).

    order : str, "t-f" (default)
        Order in which to perform interpolation and optional smoothing.
        For example, ``"t-f-s"`` means that interpolation across the OFDM symbols
        is performed first (``"t"``: time), followed by interpolation across the
        sub-carriers (``"f"``: frequency), and finally smoothing across the
        receive antennas (``"s"``: space).
        Similarly, ``"f-t"`` means interpolation across the sub-carriers followed
        by interpolation across the OFDM symbols and no spatial smoothing.
        The spatial covariance matrix (``cov_time_space``) is only required when
        spatial smoothing is requested.
        Time and frequency interpolation are not optional to ensure that a channel
        estimate is computed for all resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `tf.complex`
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `tf.complex`
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, `tf.float`
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern, cov_mat_time, cov_mat_freq,
                    cov_mat_space=None, order='t-f'):

        super().__init__()

        # Check the specified order
        order = order.split('-')
        assert 2 <= len(order) <= 3, "Invalid order for interpolation."
        spatial_smoothing = False
        freq_smoothing = False
        time_smoothing = False
        for o in order:
            assert o in ('s', 'f', 't'), f"Uknown dimension {o}"
            if o == 's':
                assert not spatial_smoothing,\
                    "Spatial smoothing can be specified at most once"
                spatial_smoothing = True
            elif o == 't':
                assert not time_smoothing,\
                    "Temporal interpolation can be specified once only"
                time_smoothing = True
            elif o == 'f':
                assert not freq_smoothing,\
                    "Frequency interpolation can be specified once only"
                freq_smoothing = True
        if spatial_smoothing:
            assert cov_mat_space is not None,\
                "A spatial covariance matrix is required for spatial smoothing"
        assert freq_smoothing, "Frequency interpolation is required"
        assert time_smoothing, "Time interpolation is required"

        self._order = order
        self._num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
        self._num_effective_subcarriers =pilot_pattern.num_effective_subcarriers

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask(pilot_pattern)

        # Build indices for mapping channel estimates and
        # error variances that are given as input to a
        # resource grid
        num_pilots = pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices = self._build_inputs2rg_indices(pilot_mask,
                                                             num_pilots)
        self._inputs_to_rg_indices = tf.cast(inputs_to_rg_indices, tf.int32)

        # 1D interpolator according to requested order
        # Interpolation is always performed along the inner dimension.
        interpolators = []
        # Masks for masking error variances that were not updated
        err_var_masks = []
        # breakpoint()
        for i, o in enumerate(order):
            # Is it the last one?
            last_step = i == len(order)-1
            # Frequency
            if o == "f":
                interpolator = LMMSEInterpolator1D(pilot_mask, cov_mat_freq,
                                                        last_step=last_step)
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                err_var_mask = tf.cast(pilot_mask == 1,
                                        tf.float32 if cov_mat_freq.dtype == np.complex64 else tf.float64)
            # Time
            elif o == 't':
                pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
                interpolator = LMMSEInterpolator1D(pilot_mask, cov_mat_time,
                                                        last_step=last_step)
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
                err_var_mask = tf.cast(pilot_mask == 1,
                                            tf.float32 if cov_mat_freq.dtype == np.complex64 else tf.float64)
            # Space
            else:
                interpolator = SpatialChannelFilter(cov_mat_space,
                                                    last_step=last_step)
                err_var_mask = tf.cast(pilot_mask == 1,
                                            tf.float32 if cov_mat_freq.dtype == np.complex64 else tf.float64)
            interpolators.append(interpolator)
            err_var_masks.append(err_var_mask)
        self._interpolators = interpolators
        self._err_var_masks = err_var_masks

    def _build_pilot_mask(self, pilot_pattern):
        """
        Build for every transmitter and stream a pilot mask indicating
        which REs are allocated to pilots, data, or not used.
        # 0 -> Data
        # 1 -> Pilot
        # 2 -> Not used
        """

        mask = pilot_pattern.mask
        pilots = pilot_pattern.pilots
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols,
                                num_effective_subcarriers], int)
        for tx,st in itertools.product( range(num_tx),
                                        range(num_streams_per_tx)):
            pil_index = 0
            for sb,sc in itertools.product( range(num_ofdm_symbols),
                                            range(num_effective_subcarriers)):
                if mask[tx,st,sb,sc] == 1:
                    if np.abs(pilots[tx,st,pil_index]) > 0.0:
                        pilot_mask[tx,st,sb,sc] = 1
                    else:
                        pilot_mask[tx,st,sb,sc] = 2
                    pil_index += 1

        return pilot_mask

    def _build_inputs2rg_indices(self, pilot_mask, num_pilots):
        """
        Builds indices for mapping channel estimates and
        error variances that are given as input to a
        resource grid
        """

        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros([num_tx, num_streams_per_tx,
                                         num_pilots, 4], int)
        for tx,st in itertools.product( range(num_tx),
                                        range(num_streams_per_tx)):
            pil_index = 0 # Pilot index for this stream
            for sb,sc in itertools.product( range(num_ofdm_symbols),
                                            range(num_effective_subcarriers)):
                if pilot_mask[tx,st,sb,sc] == 0:
                    continue
                if pilot_mask[tx,st,sb,sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                pil_index += 1

        return inputs_to_rg_indices

    def _update_pilot_mask_interp(self, pilot_mask):
        """
        Update the pilot mask to label the resource elements for which the
        channel was interpolated.
        """

        interpolated = np.any(pilot_mask == 1, axis=-1, keepdims=True)
        pilot_mask = np.where(interpolated, 1, pilot_mask)

        return pilot_mask

    def __call__(self, h_hat, err_var):
        # print('-----lmmse_interpolator 被执行了')

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, num_pilots]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, num_pilots]
        # breakpoint()
        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        num_ofdm_symbols = self._num_ofdm_symbols
        num_effective_subcarriers = self._num_effective_subcarriers

        # For some estimator, err_var might not have the same shape
        # as h_hat
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))

        # Mapping the channel estimates and error variances to a resource grid
        # all : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_effective_subcarriers]
        h_hat = tf.transpose(h_hat, [3, 4, 5, 0, 1, 2])
        err_var = tf.transpose(err_var, [3, 4, 5, 0, 1, 2])
        h_hat = tf.scatter_nd(self._inputs_to_rg_indices, h_hat,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        err_var = tf.scatter_nd(self._inputs_to_rg_indices, err_var,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        h_hat = tf.transpose(h_hat, [4, 5, 6, 0, 1, 2, 3])
        err_var = tf.transpose(err_var, [4, 5, 6, 0, 1, 2, 3])

        # Interpolation
        # Performed according to the requested order. Transpose are used as
        # 1D interpolation is performed along the inner axis.
        # breakpoint()
        items = zip(self._order, self._interpolators, self._err_var_masks)
        for o,interp,err_var_mask in items:
            # Frequency
            if o == 'f':
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_ofdm_symbols, num_effective_subcarriers]
                # breakpoint()
                h_hat, err_var = interp(h_hat, err_var)
                # interpolator = LMMSEInterpolator1D(pilot_mask, cov_mat_freq,
                #                                         last_step=last_step)
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask
            # Time
            elif o == 't':
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_effective_subcarriers, num_ofdm_symbols]
                h_hat = tf.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
                err_var = tf.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
                # breakpoint()
                h_hat, err_var = interp(h_hat, err_var)
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_ofdm_symbols, num_effective_subcarriers]
                h_hat = tf.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
                err_var = tf.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask
            # Space
            elif o == 's':
                # [batch_size, num_rx, num_tx, num_streams_per_tx,
                #      num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
                h_hat = tf.transpose(h_hat, [0, 1, 3, 4, 5, 6, 2])
                err_var = tf.transpose(err_var, [0, 1, 3, 4, 5, 6, 2])
                h_hat, err_var = interp(h_hat, err_var)
                # [batch_size, num_rx, num_tx, num_streams_per_tx,
                #      num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
                h_hat = tf.transpose(h_hat, [0, 1, 6, 2, 3, 4, 5])
                err_var = tf.transpose(err_var, [0, 1, 6, 2, 3, 4, 5])
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask

        return h_hat, err_var

#######################################################
# Utilities
#######################################################

def tdl_freq_cov_mat(model, subcarrier_spacing, fft_size, delay_spread,
                        precision=None):
    # pylint: disable=line-too-long
    r"""
    Computes the frequency covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    The channel frequency covariance matrix :math:`\mathbf{R}^{(f)}` of a TDL channel model is

    .. math::
        \mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M

    where :math:`M` is the FFT size, :math:`L` is the number of paths for the selected TDL model,
    :math:`P_\ell` and :math:`\tau_\ell` are the average power and delay for the
    :math:`\ell^{\text{th}}` path, respectively, and :math:`\Delta_f` is the sub-carrier spacing.

    Input
    ------
    model : "A" | "B" | "C" | "D" | "E"
        TDL model for which to return the covariance matrix

    subcarrier_spacing : `float`
        Sub-carrier spacing [Hz]

    fft_size : `int`
        FFT size

    delay_spread : `float`
        Delay spread [s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
        cov_mat : [fft_size, fft_size], `tf.complex`
            Channel frequency covariance matrix
    """
    if precision is None:
        precision = config.precision
    cdtype =dtypes[precision]['tf']['cdtype']

    #
    # Load the power delay profile
    #

    # Set the file from which to load the model
    assert model in ('A', 'B', 'C', 'D', 'E'), "Invalid TDL model"
    if model == 'A':
        parameters_fname = "TDL-A.json"
    elif model == 'B':
        parameters_fname = "TDL-B.json"
    elif model == 'C':
        parameters_fname = "TDL-C.json"
    elif model == 'D':
        parameters_fname = "TDL-D.json"
    else: # 'E'
        parameters_fname = "TDL-E.json"
    source = files(models).joinpath(parameters_fname)
    # pylint: disable=unspecified-encoding
    with open(source) as parameter_file:
        params = json.load(parameter_file)
    # LoS scenario ?
    los = bool(params['los'])
    # Retrieve power and delays
    delays = np.array(params['delays'])*delay_spread
    mean_powers = np.power(10.0, np.array(params['powers'])/10.0)

    if los:
        # Add the power of the specular and non-specular component of
        # the first path
        mean_powers[0] = mean_powers[0] + mean_powers[1]
        mean_powers = np.concatenate([mean_powers[:1], mean_powers[2:]], axis=0)
        # The first two paths have 0 delays as they correspond to the
        # specular and reflected components of the first path.
        delays = delays[1:]

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    #
    # Build frequency covariance matrix
    #

    n = np.arange(fft_size)
    p = -2.*np.pi*subcarrier_spacing*n
    p = np.expand_dims(p, axis=0)
    delays = np.expand_dims(delays, axis=1)
    p = p*delays
    p = np.exp(1j*p)
    p = np.expand_dims(p, axis=-1)
    cov_mat = np.matmul(p, np.transpose(np.conj(p), [0, 2, 1]))
    mean_powers = np.expand_dims(mean_powers, axis=(1,2))
    cov_mat = np.sum(mean_powers*cov_mat, axis=0)

    return tf.cast(cov_mat, cdtype)

def tdl_time_cov_mat(model, speed, carrier_frequency, ofdm_symbol_duration,
        num_ofdm_symbols, los_angle_of_arrival=PI/4., precision=None):
    # pylint: disable=line-too-long
    r"""
    Computes the time covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    For non-line-of-sight (NLoS) model, the channel time covariance matrix
    :math:`\mathbf{R^{(t)}}` of a TDL channel model is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)

    where :math:`J_0` is the zero-order Bessel function of the first kind,
    :math:`\Delta_t` the duration of an OFDM symbol, and :math:`\nu` the Doppler
    spread defined by

    .. math::
        \nu = 2 \pi \frac{v}{c} f_c

    where :math:`v` is the movement speed, :math:`c` the speed of light, and
    :math:`f_c` the carrier frequency.

    For line-of-sight (LoS) channel models, the channel time covariance matrix
    is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}

    where :math:`\alpha_{\text{LoS}}` is the angle-of-arrival for the LoS path,
    :math:`P_{\text{NLoS}}` the total power of NLoS paths, and
    :math:`P_{\text{LoS}}` the power of the LoS path. The power delay profile
    is assumed to have unit power, i.e., :math:`P_{\text{NLoS}} + P_{\text{LoS}} = 1`.

    Input
    ------
    model : "A" | "B" | "C" | "D" | "E"
        TDL model for which to return the covariance matrix

    speed : `float`
        Speed [m/s]

    carrier_frequency : `float`
        Carrier frequency [Hz]

    ofdm_symbol_duration : `float`
        Duration of an OFDM symbol [s]

    num_ofdm_symbols : `int`
        Number of OFDM symbols

    los_angle_of_arrival : `float`, (default pi/5)
        Angle-of-arrival for LoS path [radian]. Only used with LoS models.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
        cov_mat : [num_ofdm_symbols, num_ofdm_symbols], `tf.complex`
            Channel time covariance matrix
    """
    if precision is None:
        precision = config.precision
    cdtype =dtypes[precision]['tf']['cdtype']

    # Doppler spread
    doppler_spread = 2.*PI*speed/SPEED_OF_LIGHT*carrier_frequency

    #
    # Load the power delay profile
    #

    # Set the file from which to load the model
    assert model in ('A', 'B', 'C', 'D', 'E'), "Invalid TDL model"
    if model == 'A':
        parameters_fname = "TDL-A.json"
    elif model == 'B':
        parameters_fname = "TDL-B.json"
    elif model == 'C':
        parameters_fname = "TDL-C.json"
    elif model == 'D':
        parameters_fname = "TDL-D.json"
    else: # 'E'
        parameters_fname = "TDL-E.json"
    source = files(models).joinpath(parameters_fname)
    # pylint: disable=unspecified-encoding
    with open(source) as parameter_file:
        params = json.load(parameter_file)
    # LoS scenario ?
    los = bool(params['los'])
    # Retrieve power and delays
    mean_powers = np.power(10.0, np.array(params['powers'])/10.0)

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    if los:
        los_power = mean_powers[0]
        nlos_power = np.sum(mean_powers[1:])
    else:
        nlos_power = np.sum(mean_powers)

    #
    # Build time covariance matrix
    #

    indices = np.arange(num_ofdm_symbols)
    s1 = np.expand_dims(indices, axis=1)
    s2 = np.expand_dims(indices, axis=0)
    exp = doppler_spread*ofdm_symbol_duration*(s1-s2)
    cov_mat_nlos = jv(0.0, exp)*nlos_power
    if los:
        cov_mat_los = np.exp(1j*exp*np.cos(los_angle_of_arrival))*los_power
        cov_mat = cov_mat_nlos+cov_mat_los
    else:
        cov_mat = cov_mat_nlos

    return tf.cast(cov_mat, cdtype)



