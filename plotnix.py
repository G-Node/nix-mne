import sys
import matplotlib.pyplot as plt
import nixio as nix


fname = sys.argv[1]
nixfile = nix.File(fname, mode=nix.FileMode.ReadOnly)
block = nixfile.blocks[0]

nda = len(block.data_arrays)
nda = min(nda, 3)
mtag = block.multi_tags[0]

for idx, da in enumerate(block.data_arrays):
    print(f"Plotting signal {idx}")
    plt.subplot(nda, 1, idx+1)
    plt.plot(da[:])

    # for npos in range(len(mtag.positions)):
    #     mtagdata = mtag.retrieve_data(npos, idx)
    #     plt.plot(mtagdata[:])

    mtagdata = mtag.retrieve_data(0, idx)
    print(mtagdata)

    if idx == nda-1:
        break

plt.show()

nixfile.close()
