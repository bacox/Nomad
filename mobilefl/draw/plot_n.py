import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
if __name__ == "__main__":
    width = 4.2
    height = 2.8
    d90_time = {}
    d90_updates = {}
    d95_time = {}
    d95_updates = {}
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    d90_time[16] = 13521
    d90_updates[16] = 1628
    d95_time[16] = 45318
    d95_updates[16] = 5416
    d90_time[24] = 14328
    d90_updates[24] = 2529
    d95_time[24] = 29716
    d95_updates[24] = 5237
    d90_time[32] = 18217
    d90_updates[32] = 3951
    d95_time[32] = 42219
    d95_updates[32] = 9150
    d90_time[40] = 22328
    d90_updates[40] = 5444
    d95_time[40] = 50819
    d95_updates[40] = 12389
    d90_time[48] = 29319
    d90_updates[48] = 7032
    d95_time[48] = 59219
    d95_updates[48] = 14203
    d90_time[56] = 32419
    d90_updates[56] = 8654
    d90_time[64] = 40019
    d90_updates[64] = 10668
    d95_time[64] = 75719
    d95_updates[64] = 20188
    d90_time[72] = 46619
    d90_updates[72] = 12435
    d90_time[80] = 63718
    d90_updates[80] = 16991
    d95_time[80] = 115628
    d95_updates[80] = 30835
    d90_time[88] = 68218
    d90_updates[88] = 18591
    d95_time[88] = 125818
    d95_updates[88] = 33556
    d90_time[96] = 90418
    d90_updates[96] = 24111
    d90_time[104] = 111718
    d90_updates[104] = 29794
    d90_time[112] = 120418
    d90_updates[112] = 32114
    d90_time[120] = 154017
    d90_updates[120] = 41047
    f90_time = {}
    f90_updates = {}
    f90_time[120] = 328919
    f90_updates[120] = 32891
    f90_time[112] = 228228
    f90_updates[112] = 22822
    f90_time[104] = 206525
    f90_updates[104] = 20650
    f90_time[96] = 190115
    f90_updates[96] = 19011
    f90_time[88] = 165265
    f90_updates[88] = 16525
    f90_time[80] = 145050
    f90_updates[80] = 14503
    f90_time[72] = 98593
    f90_updates[72] = 9857
    f90_time[64] = 100012
    f90_updates[64] = 9999
    f90_time[56] = 82598
    f90_updates[56] = 8257
    f90_time[48] = 54886
    f90_updates[48] = 5485
    f90_time[40] = 46568
    f90_updates[40] = 4653
    f90_time[32] = 23754
    f90_updates[32] = 2372
    f90_time[24] = 14502
    f90_updates[24] = 1447
    f90_time[16] = 5394
    f90_updates[16] = 535
    n90 = list(d90_updates.keys())
    n95 = list(d95_updates.keys())
    m90 = list(f90_updates.keys())
    conv_updates95 = list(d95_updates.values())
    conv_time95 = list(d95_time.values())
    conv_time90 = list(d90_time.values())
    conv_updates90 = list(d90_updates.values())
    fconv_time90 = list(f90_time.values())
    fconv_updates90 = list(f90_updates.values())
    fig1.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%dK"))
    ax1.plot(n90, np.array(conv_updates90) // 1000, label="MultiSync 90% Accuracy")
    ax1.plot(m90, np.array(fconv_updates90) // 1000, label="FedAsync 90% Accuracy")
    ax1.set_xlabel("Number of Clients")
    ax1.set_ylabel("Number of Updates")
    ax2.plot(n90, np.array(conv_time90) // 1000, label="MultiSync 90% Accuracy")
    ax2.plot(m90, np.array(fconv_time90) // 1000, label="FedAsync 90% Accuracy")
    ax2.set_xlabel("Number of Clients")
    ax2.set_ylabel("Running Time (s)")
    ax1.legend()
    ax2.legend()
    plt.show()
