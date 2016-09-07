# mt-training
Training automation for Moses-based machine translation engines

## Installation

### Requirements
* Python >= 3.5
* [https://github.com/moses-smt/mosesdecoder](Moses) (tested with release 3.0)
* [https://github.com/clab/fast_align](fast_align)

### Environment variables
`mtrain` uses the `MOSES_HOME` and `FASTALIGN_HOME` environment variables to
locate Moses and fast_align scripts and binaries, respectively. `MOSES_HOME`
should point to the base directory of your Moses installation, containing the
subdirectories `scripts`, `bin`, etc. `FASTALIGN_HOME` is where your compiled
fast_align binaries (`fastalign` and `atools`) are stored.

#### Setting environment variables temporarily
To set `MOSES_HOME` and `FASTALIGN_HOME` for the duration of your shell session,
type

```bash
export MOSES_HOME=/path/to/moses
export FASTALIGN_HOME=/path/to/fastalign/bin
```

#### Setting environment variables permanently
If you want the environment variables to be loaded automatically for each of
your shell sessions, simply add the two export statements above to your
`~/.bashrc` or `~/.bash_profile`.

### Installation using [https://pypi.python.org/pypi/pip](pip)
```sh
# downloads the code
git clone https://gitlab.cl.uzh.ch/laeubli/mtrain.git
cd mtrain
# installs mtrain
pip install .
```
After installation, the main binaries should be available on your system. Just
type `mtrain` or `mtrans` into your console to make sure.

### Testing (optional)
In the base directory of the downloaded code (see above), type
```sh
python setup.py test
```
Make sure that the environment variables (see above) are set before running the
tests.
