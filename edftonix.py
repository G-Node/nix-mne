import os
import sys
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import mne
import nixio as nix


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an EDF filename as argument")
        sys.exit(1)

    efname = sys.argv[1]
    root, ext = os.path.splitext(efname)
    nfname = root + os.path.extsep + "nix"
    print(f"Converting '{efname}' to NIX")
    # stim_channel=False marks the last channel as "EDF Annotations"
    ef = mne.io.read_raw_edf(efname, preload=True, stim_channel=False)
    efinfo = ef.info

    # data and times
    data = ef.get_data()
    time = ef.times

    nchan = efinfo["nchan"]
    print(f"Found {nchan} channels with {ef.n_times} samples per channel")

    # Create NIX file
    nf = nix.File(nfname, nix.FileMode.Overwrite)

    # Write Data to NIX
    block = nf.create_block("EEG Data Block", "Recording")
    da = block.create_data_array("EEG Data", "Raw Data", data=data)
    da.unit = "V"

    for dimlen in data.shape:
        if dimlen == nchan:
            # channel labels: SetDimension
            da.append_set_dimension(labels=ef.ch_names)
        elif dimlen == ef.n_times:
            # times: RangeDimension
            # NOTE: EDF always uses seconds
            da.append_range_dimension(ticks=time, label="time", unit="s")

    # Write metadata to NIX
    # info dictionary
    sec = nf.create_section("Info", "File Metadata")
    for k, v in ef.info.items():
        if v is None:
            continue
        if isinstance(v, Iterable):
            if not len(v):
                continue
            # check element type
            # if isinstance(v[0], dict):
            # Create a new Section to hold the metadata found in the
            # dictionary
            #     create_sub_section(v)

        print(f"Creating metadata key {k} with value {v}")
        sec.create_property(k, v)

    nf.close()
    print(f"Created NIX file at '{nfname}'")
    print("Done")

# for idx in range(5):
#     channel_idx = idx
#     signal = da[channel_idx]
#     tdim = da.dimensions[1]
#     datadim = da.dimensions[0]

#     plt.plot(tdim.ticks, da[channel_idx], label=datadim.labels[idx])
#     xlabel = f"({tdim.unit})"
#     plt.xlabel(xlabel)
#     ylabel = f"{datadim.labels[channel_idx]} ({da.unit})"
#     plt.ylabel(ylabel)
#     plt.legend()
# plt.show()
