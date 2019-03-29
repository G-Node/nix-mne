"""
readrawnix.py

Usage:
  python readrawnix.py <nixfile>

Arguments:
  nixfile   Path to the NIX file to read.


(Requires Python 3)

Command line script for reading NIX files into an MNE structure (mne-python).
NIX file should have been created using the mnetonix.py script/module.  This
reader expects certain objects relationships and names to exist in order to
load all data and metadata successfully.  Refer to the "NIX Format Layout" in
the mnetonix.py module for details.

To include in a script, call the 'importnix()' and provide a NIX filename.
"""
import os
import sys
import numpy as np
import nixio as nix
import mne


DATA_BLOCK_NAME = "EEG Data Block"
DATA_BLOCK_TYPE = "Recording"
RAW_DATA_GROUP_NAME = "Raw Data Group"
RAW_DATA_GROUP_TYPE = "EEG Channels"
RAW_DATA_TYPE = "Raw Data"


typemap = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "tuple": tuple,
    "list": list,
    "numpy.float64": np.float64,
    "numpy.ndarray": np.array,
}


def convert_prop_type(prop):
    pt = prop.type[8:-2]
    pv = prop.values
    if len(pv) == 1:
        pv = pv[0]

    return typemap[pt](pv)


def md_to_dict(section):
    sdict = dict()
    for prop in section.props:
        sdict[prop.name] = convert_prop_type(prop)

    if section.type[8:-2] == "mne.transforms.Transform":
        to = sdict["to"]
        fro = sdict["from"]
        trans = sdict["trans"]
        trans = section.referring_data_arrays[0][:]
        return mne.Transform(fro=fro, to=to, trans=trans)

    for sec in section.sections:
        if sec.name == "chs":
            # make a list of dictionaries for the channels
            chlist = list()
            for chsec in sec.sections:
                chlist.append(md_to_dict(chsec))
                sdict[sec.name] = chlist
        else:
            sdict[sec.name] = md_to_dict(sec)

    return sdict


def merge_data_arrays(arrays):
    rows = [a[:] for a in arrays]
    return np.array(rows)


def create_mne_annotations(mtags):
    onset = list()
    duration = list()
    description = list()
    for mtag in mtags:
        posshape = mtag.positions.shape
        if len(posshape) == 1:
            onset.extend(mtag.positions)
            duration.extend(mtag.extents)
        else:
            onset.extend([p[1] for p in mtag.positions])
            duration.extend([e[1] for e in mtag.extents])

        description.extend(mtag.positions.dimensions[0].labels)

    return mne.Annotations(onset=onset,
                           duration=duration,
                           description=description)


def import_nix(nixfilename):
    """
    Import a NIX file (generated with mnetonix.py) into an MNE Raw structure.

    :param nixfilename: Path to the NIX file to be loaded.
    :rtype: mne.io.RawArray
    """
    nixfile = nix.File(nixfilename, mode=nix.FileMode.ReadOnly)

    # root, ext = os.path.splitext(nixfilename)
    # bvfilename = root + os.extsep + "vhdr"
    # bvfile = mne.io.read_raw_brainvision(bvfilename, stim_channel=False)

    # Create MNE Info object
    infosec = nixfile.sections["Info"]
    nchan = infosec["nchan"]
    sfreq = infosec["sfreq"]
    info = mne.create_info(nchan, sfreq)

    nixinfodict = md_to_dict(infosec)
    info.update(nixinfodict)

    # Read raw data into MNE objects
    datagroup = nixfile.blocks[DATA_BLOCK_NAME].groups[RAW_DATA_GROUP_NAME]
    if len(datagroup.data_arrays) > 1:
        # Data split: One DataArray per channel.  Merging
        nixrawdata = merge_data_arrays(datagroup.data_arrays)
    else:
        nixrawdata = datagroup.data_arrays[0][:]

    # Create MNE RawArray
    mnerawdata = mne.io.RawArray(nixrawdata, info)

    # Add annotations: Stimuli from MultiTags
    mtags = datagroup.multi_tags
    annotations = create_mne_annotations(mtags)

    mnerawdata.set_annotations(annotations)

    nixfile.close()

    return mnerawdata


def main():
    if len(sys.argv) < 2:
        print("Please provide either a NIX filename as the first argument")
        sys.exit(1)

    nixfilename = sys.argv[1]
    import_nix(nixfilename)


if __name__ == "__main__":
    main()
