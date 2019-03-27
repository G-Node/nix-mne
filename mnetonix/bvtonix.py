import os
import sys
import mne
import mnetonix as m2n


def main():
    if len(sys.argv) < 2:
        print("Please provide BrainVision vhdr filename as argument")
        sys.exit(1)

    bvfname = sys.argv[1]
    locname = None
    if len(sys.argv) > 2:
        locname = sys.argv[2]
        locname = os.path.abspath(locname)
    root, ext = os.path.splitext(bvfname)
    nfname = root + os.path.extsep + "nix"
    print(f"Converting '{bvfname}' to NIX")
    bvf = mne.io.read_raw_brainvision(bvfname)

    m2n.write_raw_mne(nfname, bvf)
    bvf.close()


if __name__ == "__main__":
    main()
