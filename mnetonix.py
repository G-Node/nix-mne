"""
mnetonix.py

Usage:
  python mnetonix.py <datafile> <montage>

datafile: Either an EDF file or a BrainVision header file (vhdr).
montage: Any format montage file supported by MNE.

(Requires Python 3)

Command line script for reading EDF and BrainVision files using MNE
(mne-python) and storing the data and metadata into a NIX file.  Supports
reading montage files for recording channel locations.

NIX Format layout:

Data:
Raw Data are stored in either a single 2-dimensional DataArray or a collection
of DataArrays (one per recording channel).  The latter makes tagging easier
since MultiTag positions and extents don't need to specify every channel they
reference.  However, creating multiple DataArrays makes file sizes much
bigger.

Stimuli:
MNE provides stimulus information through the Raw.annotations dictionary.
Onsets correspond to the 'positions' array and durations correspond to the
'extents' array of the "Stimuli" MultiTag.

Metadata:
MNE collects metadata into a (nested) dictionary (Raw.info).  All non-empty
keys are converted into Properties in NIX.  The nested structure of the
dictionary is replicated in NIX by creating child Sections, starting with one
root section with name "Info".

"""
import sys
import os
from collections.abc import Iterable, Mapping
import mne
import matplotlib.pyplot as plt
import numpy as np
import nixio as nix


DATA_BLOCK_NAME = "EEG Data Block"
DATA_BLOCK_TYPE = "Recording"
RAW_DATA_GROUP_NAME = "Raw Data Group"
RAW_DATA_GROUP_TYPE = "EEG Channels"
RAW_DATA_TYPE = "Raw Data"


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


def create_md_tree(section, values, block):
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
                da = block.create_data_array(k, "Multidimensional Metadata",
                                             data=v)
                for _ in range(ndim):
                    da.append_set_dimension()
                section.create_property(k, da.id)
                da.metadata = section
                continue
            # check element type
            if isinstance(v, Mapping):
                # Create a new Section to hold the metadata found in the
                # dictionary
                subsec = section.create_section(k, str(v.__class__))
                create_md_tree(subsec, v, block)
                continue
            elif isinstance(v[0], Mapping):
                # Create multiple new Sections to hold the metadata found in
                # each nested dictionary
                for idx, subd in enumerate(v):
                    secname = f"{k}-{idx}"
                    subsec = section.create_section(secname, str(v.__class__))
                    create_md_tree(subsec, subd, block)
                continue

        try:
            section.create_property(k, v)
        except TypeError:
            # inconsistent iterable types: upgrade to floats
            section.create_property(k, [float(vi) for vi in v])


def write_single_da(mneraw, block):
    # data and times
    data = mneraw.get_data()
    time = mneraw.times

    nchan = mneraw.info["nchan"]
    print(f"Found {nchan} channels with {mneraw.n_times} samples per channel")

    da = block.create_data_array("EEG Data", RAW_DATA_TYPE, data=data)
    block.groups[RAW_DATA_GROUP_NAME].data_arrays.append(da)
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
        da = block.create_data_array(chname, RAW_DATA_TYPE, data=chandata)
        block.groups[RAW_DATA_GROUP_NAME].data_arrays.append(da)
        da.unit = "V"

        # times: RangeDimension
        # NOTE: EDF always uses seconds
        da.append_range_dimension(ticks=time, label="time", unit="s")


def write_stim_tags(mneraw, block):
    # check dimensionality of data
    datashape = block.groups[RAW_DATA_GROUP_NAME].data_arrays[0].shape
    stimuli = mneraw.annotations
    labels = stimuli.description

    ndim = len(datashape)
    if ndim == 1:
        positions = stimuli.onset
        extents = stimuli.duration
    else:
        channelextent = mneraw.info["nchan"] - 1
        positions = [(0, p) for p in stimuli.onset]
        extents = [(channelextent, e) for e in stimuli.duration]

    posda = block.create_data_array("Stimuli onset", "Stimuli Positions",
                                    data=positions)
    posda.append_set_dimension(labels=labels.tolist())

    extda = block.create_data_array("Stimuli Durations", "Stimuli Extents",
                                    data=extents)
    extda.append_set_dimension(labels=labels.tolist())

    for _ in range(ndim-1):
        # extra set dimensions for any extra data dimensions (beyond the first)
        posda.append_set_dimension()
        extda.append_set_dimension()

    stimmtag = block.create_multi_tag("Stimuli", "EEG Stimuli",
                                      positions=posda)
    stimmtag.extents = extda
    block.groups[RAW_DATA_GROUP_NAME].multi_tags.append(stimmtag)

    for da in block.groups[RAW_DATA_GROUP_NAME].data_arrays:
        if da.type == RAW_DATA_TYPE:
            stimmtag.references.append(da)


def write_raw_mne(nfname, mneraw, split_data_channels=False):
    mneinfo = mneraw.info
    extrainfo = mneraw._raw_extras

    # Create NIX file
    nf = nix.File(nfname, nix.FileMode.Overwrite)

    # Write Data to NIX
    block = nf.create_block(DATA_BLOCK_NAME, DATA_BLOCK_TYPE,
                            compression=nix.Compression.DeflateNormal)
    block.create_group(RAW_DATA_GROUP_NAME, RAW_DATA_GROUP_TYPE)

    if split_data_channels:
        write_multi_da(mneraw, block)
    else:
        write_single_da(mneraw, block)

    if mneraw.annotations:
        write_stim_tags(mneraw, block)

    # Write metadata to NIX
    # info dictionary
    infomd = nf.create_section("Info", "File metadata")
    create_md_tree(infomd, mneinfo, block)
    # extras
    if len(extrainfo) > 1:
        for idx, emd_i in enumerate(extrainfo):
            extrasmd = nf.create_section(f"Extras-{idx}",
                                         "Raw Extras metadata")
            create_md_tree(extrasmd, emd_i, block)
    elif extrainfo:
        extrasmd = nf.create_section("Extras", "Raw Extras metadata")
        create_md_tree(extrasmd, extrainfo[0], block)

    # all done
    nf.close()
    print(f"Created NIX file at '{nfname}'")
    print("Done")


def main():
    if len(sys.argv) < 2:
        print("Please provide either a BrainVision vhdr or "
              "an EDF filename as the first argument")
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

    write_raw_mne(nfname, mneraw, True)

    nfname = root + "-oneda" + os.path.extsep + "nix"
    write_raw_mne(nfname, mneraw, False)

    mneraw.close()


if __name__ == "__main__":
    main()
