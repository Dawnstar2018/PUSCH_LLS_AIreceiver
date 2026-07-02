import os
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={os.environ["CONDA_PREFIX"]}'
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from scipy.special import jv  # 零阶贝塞尔函数
# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    import os
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

sionna.phy.config.seed = 42 # Set seed for reproducible results

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.channel import AWGN, RayleighBlockFading, OFDMChannel, \
                               TimeChannel, time_lag_discrete_time_channel,AWGN_MIMO,AWGN_MIMO_time
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa, RMa, CDL
from sionna.phy.utils import compute_ber, ebnodb2no, sim_ber
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import pickle
np.random.seed(42)
# from main_e2e import Model
class Model(Block):
    """Simulate PUSCH transmissions over a 3GPP 38.901 model

    This model runs BER simulations for a multiuser MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).
    You can chosse to run simulations in either time ("time") or frequency ("freq")
    domains and configure different user speeds.

    Parameters
    ----------
    scenario : str, one of ["umi", "uma", "rma"]
        3GPP 38.901 channel model to be used

    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated

    domain :  str, one of ["freq", "time"]
        Domain in which the simulations are carried out.
        Time domain modelling is typically more complex but allows modelling
        of realistic effects such as inter-symbol interference of subcarrier
        interference due to very high speeds.

    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.

    speed: float
        User speed (m/s)

    Input
    -----
    batch_size : int
        Number of simultaneously simulated slots

    ebno_db : float
        Signal-to-noise-ratio

    Output
    ------
    b : [batch_size, num_tx, tb_size], torch.float
        Transmitted information bits

    b_hat : [batch_size, num_tx, tb_size], torch.float
        Decoded information bits
    """
    def __init__(self,
                 scenario,    # "umi", "uma", "rma"
                 perfect_csi, # bool
                 domain,      # "freq", "time"
                 detector,    # "lmmse", "kbest"
                 chennel_est_model, #"ls" ,"mmse"
                 speed        # float
                ):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._domain = domain
        self._speed = speed

        self._carrier_frequency = 3.5e9
        self._subcarrier_spacing = 30e3
        self._num_tx = 1
        self._num_tx_ant = 2
        self._num_layers = 2
        self._num_rx_ant = 4
        self._mcs_index = 14
        self._mcs_table = 1
        self._num_prb = 10
        self._cdl_model = 'C'
        self._delay_spread = 100e-9
        # self._channel_est_mode = chennel_est_model

        # Create PUSCHConfigs

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 0
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 1
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table
        pusch_config.dmrs.show()
        pusch_config.tb.show()
        print("TB_SIZE:",pusch_config.tb_size)
        print('available RE',pusch_config.available_re)
        print('dmrs symbol idx',pusch_config.dmrs_symbol_indices)
        
        # Create PUSCHConfigs for the other transmitters by cloning of the first PUSCHConfig
        # and modifying the used DMRS ports.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_tx):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i*self._num_layers, (i+1)*self._num_layers))
            print('dmrs_port_set',i,':',pc.dmrs.dmrs_port_set)
            pusch_configs.append(pc)
        # breakpoint()
        # Create PUSCHTransmitter
        self._pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=self._domain)
        print('------------')
        print(self._pusch_transmitter.resource_grid._pilot_pattern.mask.shape)
        print(self._pusch_transmitter.resource_grid.fft_size)
        print(self._pusch_transmitter.resource_grid._num_ofdm_symbols)
        print(self._pusch_transmitter.resource_grid.num_effective_subcarriers)

        # Create PUSCHReceiver
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._pusch_transmitter.resource_grid.bandwidth)


        rx_tx_association = np.ones([1, self._num_tx], bool)
        stream_management = StreamManagement(rx_tx_association,
                                             self._num_layers)

        assert detector in["lmmse", "kbest"], "Unsupported MIMO detector"
        if detector=="lmmse":
            detector = LinearDetector(equalizer="lmmse",
                                      output="bit",
                                      demapping_method="maxlog",
                                      resource_grid=self._pusch_transmitter.resource_grid,
                                      stream_management=stream_management,
                                      constellation_type="qam",
                                      num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        elif detector=="kbest":
            detector = KBestDetector(output="bit",
                                     num_streams=self._num_tx*self._num_layers,
                                     k=64,
                                     resource_grid=self._pusch_transmitter.resource_grid,
                                     stream_management=stream_management,
                                     constellation_type="qam",
                                     num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)

        if self._perfect_csi:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator="perfect",
                                                 l_min = self._l_min)
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator = chennel_est_model,
                                                 l_min = self._l_min)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=int(self._num_tx_ant/2),
                                 polarization="dual",
                                 polarization_type="cross",
                                 antenna_pattern="38.901",
                                 carrier_frequency=self._carrier_frequency,
                                 vertical_spacing = 0.5,
                                 horizontal_spacing = 0.5)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_rx_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency,
                                      vertical_spacing = 0.5,
                                      horizontal_spacing = 0.5)

        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "cdl":
            self._channel_model = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction="uplink",
                        min_speed=self._speed)
            
        if self._scenario == "awgn":
            # self._channel = AWGN()
            if domain=="freq":
                self._channel = AWGN_MIMO(self._num_tx_ant,self._num_rx_ant)
            elif domain=="time":
                self._channel = AWGN_MIMO_time(self._num_tx_ant,self._num_rx_ant)
        else:
            # Configure the actual channel
            if domain=="freq":
                self._channel = OFDMChannel(
                                    self._channel_model,
                                    self._pusch_transmitter.resource_grid,
                                    normalize_channel=True,
                                    return_channel=True)
            else:
                self._channel = TimeChannel(
                                    self._channel_model,
                                    self._pusch_transmitter.resource_grid.bandwidth,
                                    self._pusch_transmitter.resource_grid.num_time_samples,
                                    l_min=self._l_min,
                                    l_max=self._l_max,
                                    normalize_channel=True,
                                    return_channel=True)

    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_tx,
                                self._scenario,
                                min_ut_velocity=self._speed,
                                max_ut_velocity=self._speed)

        self._channel_model.set_topology(*topology)

    def call(self, batch_size, ebno_db):
        # self.new_topology(batch_size)#这里是什么意思

        x, b = self._pusch_transmitter(batch_size)
        # print('b.shape is',b.shape,'/n',b[0,0,0:15])
        # breakpoint()
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       self._pusch_transmitter._target_coderate,
                       self._pusch_transmitter.resource_grid)
        if self._scenario == 'awgn':
            # breakpoint()
            if self._domain == 'freq':
                y = self._channel(x,no)
            elif self._domain == 'time': 
                y = self._channel(x,ebno_db)
        else:
            # breakpoint()
            y, h = self._channel(x, no)#x=[batch,tx,tx_ant,symbok,fftsize]
        if self._perfect_csi:
            b_hat = self._pusch_receiver(y, no, h)
        else:
            
            b_hat = self._pusch_receiver(y, no)
            # print('b.shape is',b_hat.shape,'/n',b_hat[0,0,0:15])
            # breakpoint()
            a=0

        return b, b_hat
if __name__ == "__main__":
    
    PUSCH_SIMS = {
        "scenario" : ["cdl"],
        "domain" : ["time"],
        "perfect_csi" : [False],
        "detector" : ["lmmse"],
        "channel_est_mode":['mmse','ls-lin'],#['ls-lin','mmse','perfect','','',]
        # "ebno_db" : list(range(4,7,0.5)),
        "ebno_db" : np.arange(5,15,1).tolist(),
        "speed" : 3,
        "batch_size_freq" : 64,
        "batch_size_time" : 32, # Reduced batch size from time-domain modeling
        "bler" : [],
        "ber" : []
        }
    plot = True
    sim = True
    save_path = "/home/zhouxx/aiml/code/dmrs_OH_reduction/baseline_calibration/add_matlab_mmse/"
    save_file_name = 'cdl-2T4R_rank2_mcs14_10prb_ds100ns_1addsym_3p5GHZ_speed3ms_timedomain'
    print("simulation case:",save_file_name)
    # chennel_est_model = PUSCH_SIMS["channel_est_mode"]
    start = time.time()
    if sim:
        for scenario in PUSCH_SIMS["scenario"]:
            for domain in PUSCH_SIMS["domain"]:
                for perfect_csi in PUSCH_SIMS["perfect_csi"]:
                    batch_size = PUSCH_SIMS["batch_size_freq"] if domain=="freq" else PUSCH_SIMS["batch_size_time"]
                    for detector in PUSCH_SIMS["detector"]:
                        for chennel_est_model in PUSCH_SIMS["channel_est_mode"]:
                            model = Model(scenario, perfect_csi, domain, detector,chennel_est_model, PUSCH_SIMS["speed"])
                            # print('model.channel model', model._channel_model)
                            # breakpoint()
                            # model._channel_model.allocate_topology_tensors(batch_size=batch_size, num_bs=1, num_ut=model._num_tx)
                            ber, bler = sim_ber(model,
                                    PUSCH_SIMS["ebno_db"],
                                    batch_size=batch_size,
                                    max_mc_iter=30,
                                    num_target_block_errors=200
                                    )
                            PUSCH_SIMS["ber"].append(list(ber.cpu().numpy()))
                            PUSCH_SIMS["bler"].append(list(bler.cpu().numpy()))

        with open(save_path +save_file_name, 'wb') as f:
            pickle.dump(PUSCH_SIMS, f)

        PUSCH_SIMS["duration"] = time.time() - start
        # Uncomment to show precomputed results
        #PUSCH_SIMS = eval("{'scenario': ['umi'], 'domain': ['freq'], 'perfect_csi': [True, False], 'detector': ['kbest', 'lmmse'], 'ebno_db': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'speed': 3.0, 'batch_size_freq': 128, 'batch_size_time': 28, 'bler': [[0.865234375, 0.525390625, 0.236328125, 0.0703125, 0.022894965277777776, 0.0081787109375, 0.0031916920731707315, 0.0011800130208333333, 0.0007274208566108007, 0.000298828125, 0.000189453125, 0.000107421875, 4.6875e-05], [0.501953125, 0.2412109375, 0.13411458333333334, 0.06556919642857142, 0.037109375, 0.021073190789473683, 0.012251420454545454, 0.007244318181818182, 0.0038869121287128713, 0.0027646346830985913, 0.0015751008064516128, 0.0009838684538653367, 0.0007239105504587156], [0.994140625, 0.97265625, 0.849609375, 0.513671875, 0.234375, 0.115234375, 0.05126953125, 0.028878348214285716, 0.023092830882352942, 0.018694196428571428, 0.013671875, 0.012451171875, 0.013739224137931034], [0.919921875, 0.724609375, 0.533203125, 0.2802734375, 0.16536458333333334, 0.08984375, 0.056919642857142856, 0.043619791666666664, 0.035006009615384616, 0.02055921052631579, 0.017578125, 0.018465909090909092, 0.01532451923076923]], 'ber': [[0.08414149284362793, 0.03808903694152832, 0.013622879981994629, 0.00516200065612793, 0.0018940899107191297, 0.0006881306568781534, 0.0002859627328267912, 0.00012890001138051352, 9.374645169220823e-05, 3.3643484115600584e-05, 2.6883602142333983e-05, 1.2900114059448242e-05, 6.676435470581055e-06], [0.032366275787353516, 0.015960693359375, 0.009874105453491211, 0.004354306629725865, 0.00270201943137429, 0.0015692459909539473, 0.0008932893926447088, 0.0005442922765558416, 0.0002903820264457476, 0.00021878598441540356, 0.00013986518306116904, 7.878217911185171e-05, 6.43913898992976e-05], [0.15517663955688477, 0.12702298164367676, 0.08907222747802734, 0.036322832107543945, 0.015564680099487305, 0.008847415447235107, 0.005330264568328857, 0.003527675356183733, 0.0029088469112620633, 0.0025938579014369418, 0.002038750155218716, 0.0017822608351707458, 0.001927071604235419], [0.10343790054321289, 0.06611466407775879, 0.043680429458618164, 0.0217667818069458, 0.013199090957641602, 0.007306861877441406, 0.005208117621285575, 0.004094309277004666, 0.003994941711425781, 0.002383282310084293, 0.0023060985233472743, 0.002356225794011896, 0.002158962763272799]], 'duration': 4399.180883407593}")
        print("Simulation duration: {:1.2f} [h]".format(PUSCH_SIMS["duration"]/3600))
    if plot:
        with open(save_path +save_file_name, 'rb') as f:
            PUSCH_SIMS = pickle.load(f)
        plt.figure()
        plt.title(save_file_name)
        plt.xlabel("SNR (dB)")
        plt.ylabel("BLER")
        plt.grid(which="both")
        plt.xlim([PUSCH_SIMS["ebno_db"][0], PUSCH_SIMS["ebno_db"][-1]])
        plt.ylim([1e-3, 1.0])

        i = 0
        legend = []
        for scenario in PUSCH_SIMS["scenario"]:
            for domain in PUSCH_SIMS["domain"]:
                for perfect_csi in PUSCH_SIMS["perfect_csi"]:
                    for channel_est_mode in PUSCH_SIMS["channel_est_mode"]:
                        # breakpoint()
                        plt.semilogy(PUSCH_SIMS["ebno_db"], PUSCH_SIMS["bler"][i])
                        i += 1
                        csi = "PerfCSI" if perfect_csi else ""
                        det = "mmse-est" if channel_est_mode=="mmse" else "ls-est"
                        channel_type = scenario
                        legend.append(channel_type+" " +det + " " + csi)
        plt.legend(legend)
        plt.savefig(save_path+"/"+save_file_name+".png")