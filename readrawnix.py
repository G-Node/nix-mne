import os
import sys
import nixio as nix
import mne


def md_to_dict(section):
    sdict = dict()
    for prop in section.props:
        values = list(prop.values)
        if len(values) == 1:
            values = values[0]
        sdict[prop.name] = values

    for sec in section.sections:
        sdict[sec.name] = md_to_dict(sec)

    return sdict


def main():
    if len(sys.argv) < 2:
        print("Please provide either a NIX filename as the first argument")
        sys.exit(1)

    nixfilename = sys.argv[1]
    nixfile = nix.File(nixfilename, mode=nix.FileMode.ReadOnly)

    root, ext = os.path.splitext(nixfilename)
    bvfilename = root + os.extsep + "vhdr"
    bvfile = mne.io.read_raw_brainvision(bvfilename, stim_channel=False)

    # Create MNE Info object
    infosec = nixfile.sections["Info"]
    nchan = infosec["nchan"]
    sfreq = infosec["sfreq"]
    info = mne.create_info(nchan, sfreq)
    print(info)

    info.update(md_to_dict(infosec))

    print(info)
    print(bvfile.info)

    nixfile.close()


if __name__ == "__main__":
    main()
