import sys
import matplotlib.pyplot as plt
import nixio as nix


fname = sys.argv[1]
nixfile = nix.File(fname, mode=nix.FileMode.ReadOnly)
block = nixfile.blocks[0]

# plot all signals
data = block.data_arrays["EEG Data"]
time = data.dimensions[1].ticks
for idx, row in enumerate(data):
    plt.plot(time, row + idx * 0.001)

stim = block.multi_tags["Stimuli"]
for idx, p in enumerate(stim.positions):
    if stim.positions.dimensions[0].labels[idx] == "Stimulus/S  2":
        plt.plot([p, p], [0, len(data) * 0.001], "k")


plt.show()

nixfile.close()
