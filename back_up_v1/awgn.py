#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Block for simulating an AWGN channel"""

import tensorflow as tf
from sionna.phy.block import Block
from sionna.phy.utils import expand_to_rank, complex_normal
import numpy as np

class AWGN(Block):
    r"""
    Add complex AWGN to the inputs with a certain variance

    This layer blocks complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> awgn_channel = AWGN()

    Running:

    >>> # x is the channel input
    >>> # no is the noise variance
    >>> y = awgn_channel(x, no)

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x :  Tensor, tf.complex
        Channel input

    no : Scalar or Tensor, `tf.float`
        Scalar or tensor whose shape can be broadcast to the shape of ``x``.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the input.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of ``x``. This allows, e.g., adding noise of different
        variance to each example in a batch. If ``no`` has a lower rank than
        ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
        dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as ``x``, `tf.complex`
            Channel output
    """

    def __init__(self, *, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

    def call(self, x, no):

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), precision=self.precision)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, tf.rank(x), axis=-1)

        # Apply variance scaling
        no = tf.cast(no, self.rdtype)
        noise *= tf.cast(tf.sqrt(no), noise.dtype)

        # Add noise to input
        y = x + noise

        return y


#zxx 发现awgn信道仅添加噪声，无法适配多天线，所以创建下述类
class AWGN_MIMO(Block):
    def __init__(self, num_tx_ant,num_rx_ant,*, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        # self.snr = ebno

        ##随机生成矩阵，但是这个矩阵的随机性比较大，矩阵对性能的影响很大
        # random_h = tf.random.normal(shape=(self.num_rx_ant, self.num_tx_ant), mean=1.0, stddev=0.5)
        # # breakpoint()
        # # 核心：转成Python列表，再用tf.constant，和你固定H语法一模一样
        # self.H = tf.constant(random_h.numpy().tolist(), dtype=self.cdtype)
        #先使用固定的矩阵进行测试
        self.H = tf.constant([
            [1.0, 0.5],
            [0.5, 1.0],
            [1.2, 2.0],
            [2.0, 1.2]
        ], dtype=self.cdtype)
        # self.H = tf.constant([
        #     [1.0 ,  1.0 ],
        #     [1.0 ,  1.0 ],
        #     [0.0, 1.0],
        #     [0.0 , 1.0]
        # ], dtype=self.cdtype)

    def call(self, x, no):
                # -------------------------- 1. 自动获取输入维度 --------------------------
        input_rank = tf.rank(x)
        # 强制校验输入为5维张量
        # breakpoint()
        tf.debugging.assert_equal(input_rank, 5,
                                message="输入x必须是5维张量：[B, L, Nt, S, F]")
        
        # 自动提取 发射天线数 (x的第3个维度，index=2)
        num_tx_ant = tf.shape(x)[2]
        batch_size = tf.shape(x)[0]
        # breakpoint()
        X0 = x[:, :, 0, :, :]  # [B, L, S, F] 发射天线0
        X1 = x[:, :, 1, :, :]  # [B, L, S, F] 发射天线1
        rx_antennas = []
        # 遍历 0~3 共4根接收天线
        for r_idx in range(self.num_rx_ant):
            # 取出当前接收天线对应的两路发射天线权重
            h_t0_r = self.H[r_idx,0]  # H(T0, Rr)
            h_t1_r = self.H[r_idx,1]  # H(T1, Rr)
            
            # 严格按照你的公式计算
            rx_signal = h_t0_r * X0 + h_t1_r * X1
            
            # 增加接收天线维度 [B,L,S,F] → [B,L,1,S,F]
            rx_signal = tf.expand_dims(rx_signal, axis=2)
            
            # 存入列表
            rx_antennas.append(rx_signal)

        # 3. 拼接4根接收天线 → 最终MIMO信号 [B,L,4,S,F]
        x_rx = tf.concat(rx_antennas, axis=2)

        # -------------------------- 4. 原版AWGN噪声逻辑 (纯TF) --------------------------
        # 生成与x_rx同形状的复高斯噪声
        noise = complex_normal(tf.shape(x_rx), precision=self.precision)
        
        # 噪声功率处理（完全兼容Sionna官方逻辑）
        if not isinstance(no, tf.Tensor):
            no = tf.convert_to_tensor(no, dtype=self.rdtype)
        no = expand_to_rank(no, tf.rank(x_rx), axis=-1)
        no = tf.cast(no, self.cdtype)
        # breakpoint()
        # 叠加噪声
        # breakpoint()
        y = x_rx + noise * tf.math.sqrt(no)

        return y

class AWGN_MIMO_time(Block):
    def __init__(self, num_tx_ant,num_rx_ant,*, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        

        ##随机生成矩阵，但是这个矩阵的随机性比较大，矩阵对性能的影响很大
        # random_h = tf.random.normal(shape=(self.num_rx_ant, self.num_tx_ant), mean=1.0, stddev=0.5)
        # # breakpoint()
        # # 核心：转成Python列表，再用tf.constant，和你固定H语法一模一样
        # self.H = tf.constant(random_h.numpy().tolist(), dtype=self.cdtype)
        #先使用固定的矩阵进行测试
        self.H = tf.constant([
            [1.0, 0.5],
            [0.5, 1.0],
            [1.2, 2.0],
            [2.0, 1.2]
        ], dtype=self.cdtype)
    
    ##时域信道加噪声
    def call(self, x, no):
        # breakpoint()
        num_rx = 1
        [batch,num_tx,num_tx_ante,ofdm_time_symbol]=x.shape
        x_rx = []
        for i in range(self.num_rx_ant):
            if num_tx_ante == 1:
                x_rx[:,:,i,:] = x
            elif num_tx_ante == 2:
                x_rx_temp = x[:, :, 0:1, :]*self.H[i,0] + x[:, :, 1:2, :]*self.H[i,1]
                x_rx.append(x_rx_temp)
            else:
                print('error:awgn channel not support the rx antenna number')
        x_rx = tf.concat(x_rx, axis=2)
        # rand_np_1 = np.random.uniform(0, 1, size=x_rx.shape)
        # rand_np_2 = np.random.uniform(0, 1, size=x_rx.shape)
        noise = complex_normal(tf.shape(x_rx), precision=self.precision)
        noise_power = 10**(-no/10)
        noise_power = tf.cast(noise_power,tf.float32)
        amp = tf.math.sqrt(noise_power)
        noise_real = tf.math.real(noise) * amp
        noise_imag = tf.math.imag(noise) * amp
        noise_scaled = tf.complex(noise_real, noise_imag)
        # breakpoint()
        y = x_rx + noise_scaled
        return y


            

                



