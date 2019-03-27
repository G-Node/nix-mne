import os
import sys
from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
import mne
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


def main():
    if len(sys.argv) < 2:
        print("Please provide an EDF filename as argument")
        sys.exit(1)

    efname = sys.argv[1]
    locname = None
    if len(sys.argv) > 2:
        locname = sys.argv[2]
        locname = os.path.abspath(locname)
    root, ext = os.path.splitext(efname)
    nfname = root + os.path.extsep + "nix"
    print(f"Converting '{efname}' to NIX")
    # stim_channel=False marks the last channel as "EDF Annotations"
    ef = mne.io.read_raw_edf(efname, montage=locname,
                             preload=True, stim_channel=False)
    efinfo = ef.info
    extrainfo = ef._raw_extras

    # data and times
    data = ef.get_data()
    time = ef.times

    nchan = efinfo["nchan"]
    print(f"Found {nchan} channels with {ef.n_times} samples per channel")

    # Create NIX file
    nf = nix.File(nfname, nix.FileMode.Overwrite)

    # Write Data to NIX
    block = nf.create_block("EEG Data Block", "Recording",
                            compression=nix.Compression.DeflateNormal)
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
    infomd = nf.create_section("Info", "File metadata")
    create_md_tree(infomd, efinfo)
    # extras
    if len(extrainfo) > 1:
        for idx, emd_i in enumerate(extrainfo):
            extrasmd = nf.create_section(f"Extras-{idx}",
                                         "Raw Extras metadata")
            create_md_tree(extrasmd, emd_i)
    else:
        extrasmd = nf.create_section("Extras", "Raw Extras metadata")
        create_md_tree(extrasmd, extrainfo[0])

    for sec in nf.sections:
        sec.pprint()

    # all done
    nf.close()
    ef.close()
    print(f"Created NIX file at '{nfname}'")
    print("Done")


if __name__ == "__main__":
    main()
