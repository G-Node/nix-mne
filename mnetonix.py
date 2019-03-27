import sys
import os
from collections.abc import Iterable
import mne
import matplotlib.pyplot as plt
import numpy as np
import nixio as nix


def plot_channel(data_array, index):
    signal = data_array[index]
    tdim = data_array.dimensions[1]
    datadim = data_array.dimensions[0]

    plt.plot(tdim.ticks, signal, label=datadim.labels[index])
    xlabel = f"({tdim.unit})"
    plt.xlabel(xlabel)
    ylabel = f"{datadim.labels[index]} ({data_array.unit})"
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def create_md_tree(section, values):
    if values is None:
        return
    for k, v in values.items():
        if v is None:
            continue
        if isinstance(v, Iterable):
            if not len(v):
                continue
            ndim = np.ndim(v)
            if ndim > 1:
                print(f"WARNING: Skipping metadata {k} with {ndim} dimensions")
                print(v)
                continue
            # check element type
            if isinstance(v, dict):
                # Create a new Section to hold the metadata found in the
                # dictionary
                subsec = section.create_section(k, "File Metadata")
                create_md_tree(subsec, v)
                continue
            elif isinstance(v[0], dict):
                # Create multiple new Sections to hold the metadata found in
                # each nested dictionary
                for idx, subd in enumerate(v):
                    secname = f"{k}-{idx}"
                    subsec = section.create_section(secname, "File Metadata")
                    create_md_tree(subsec, subd)
                continue

        section.create_property(k, v)


def write_single_da(mneraw, block):
    # data and times
    data = mneraw.get_data()
    time = mneraw.times

    nchan = mneraw.info["nchan"]
    print(f"Found {nchan} channels with {mneraw.n_times} samples per channel")

    da = block.create_data_array("EEG Data", "Raw Data", data=data)
    da.unit = "V"

    for dimlen in data.shape:
        if dimlen == nchan:
            # channel labels: SetDimension
            da.append_set_dimension(labels=mneraw.ch_names)
        elif dimlen == mneraw.n_times:
            # times: RangeDimension
            # NOTE: EDF always uses seconds
            da.append_range_dimension(ticks=time, label="time", unit="s")


def write_multi_da(mneraw, block):
    data = mneraw.get_data()
    time = mneraw.times

    nchan = mneraw.info["nchan"]
    channames = mneraw.ch_names

    print(f"Found {nchan} channels with {mneraw.n_times} samples per channel")

    # find the channel dimension to iterate over it
    for idx, dimlen in enumerate(data.shape):
        if dimlen == nchan:
            chanidx = idx
            break
    else:
        raise RuntimeError("Could not find data dimension that matches number "
                           "of channels")

    for idx, chandata in enumerate(np.rollaxis(data, chanidx)):
        chname = channames[idx]
        da = block.create_data_array(chname, "Raw Data", data=chandata)
        da.unit = "V"

        # times: RangeDimension
        # NOTE: EDF always uses seconds
        da.append_range_dimension(ticks=time, label="time", unit="s")


def write_raw_mne(nfname, mneraw, split_data_channels=True):
    mneinfo = mneraw.info
    extrainfo = mneraw._raw_extras

    # Create NIX file
    nf = nix.File(nfname, nix.FileMode.Overwrite)

    # Write Data to NIX
    block = nf.create_block("EEG Data Block", "Recording",
                            compression=nix.Compression.DeflateNormal)

    if split_data_channels:
        write_multi_da(mneraw, block)
    else:
        write_single_da(mneraw, block)
    # Write metadata to NIX
    # info dictionary
    infomd = nf.create_section("Info", "File metadata")
    create_md_tree(infomd, mneinfo)
    # extras
    if len(extrainfo) > 1:
        for idx, emd_i in enumerate(extrainfo):
            extrasmd = nf.create_section(f"Extras-{idx}",
                                         "Raw Extras metadata")
            create_md_tree(extrasmd, emd_i)
    elif extrainfo:
        extrasmd = nf.create_section("Extras", "Raw Extras metadata")
        create_md_tree(extrasmd, extrainfo[0])

    # all done
    nf.close()
    print(f"Created NIX file at '{nfname}'")
    print("Done")


def main():
    if len(sys.argv) < 2:
        print("Please provide BrainVision vhdr filename as argument")
        sys.exit(1)

    datafilename = sys.argv[1]
    montage = None
    if len(sys.argv) > 2:
        montage = sys.argv[2]
        montage = os.path.abspath(montage)
    root, ext = os.path.splitext(datafilename)
    nfname = root + os.path.extsep + "nix"
    if ext.casefold() == ".edf".casefold():
        mneraw = mne.io.read_raw_edf(datafilename, montage=montage,
                                     preload=True, stim_channel=False)
    elif ext.casefold() == ".vhdr".casefold():
        mneraw = mne.io.read_raw_brainvision(datafilename, montage=montage,
                                             preload=True, stim_channel=False)
    else:
        raise RuntimeError(f"Unknown extension '{ext}'")
    print(f"Converting '{datafilename}' to NIX")

    write_raw_mne(nfname, mneraw)
    mneraw.close()


if __name__ == "__main__":
    main()
