# mtrain

This Python 3 package provides convenience wrappers to train (`mtrain`) and
translate with (`mtrans`) machine translation engines. Two types of engines are supported: Moses and Nematus.

Given a parallel corpus of any size, training and translation are as easy as

```sh
mkdir ~/my_engine
mtrain /path/to/my/parallel-corpus en fr --tune 1000 -o ~/my_engine
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr
```

Installation and further usage instructions are given below. To report a bug or suggest
improvements, please feel free to [open a ticket](https://github.com/ZurichNLP/mtrain/issues).

## Installation

The requirements of `mtrain` depend on the backend (either Moses or Nematus) that you would like to use. All dependencies are mandatory, except indicated otherwise.

#### Requirements for Moses backend
* Python >= 3.5
* [Moses](https://github.com/moses-smt/mosesdecoder) (tested with release 3.0). Make sure to compile with cmph (`./bjam --with-cmph=/path/to/cmph`).
* [fast_align](https://github.com/clab/fast_align)
* [MultEval](https://github.com/cidermole/multeval) (optional, only for evaluation)

#### Requirements for Nematus backend
* Python >= 3.5, and Python >= 2.7. Both versions are necessary because `Nematus` is a Python 2-only tool. To manage two versions of Python on your system, we recommend virtual environments where both `python2` and `python3` are available.
* [Moses](https://github.com/moses-smt/mosesdecoder) (tested with release 3.0). Make sure to compile with cmph (`./bjam --with-cmph=/path/to/cmph`).
* [Nematus](https://github.com/EdinburghNLP/nematus), see their Github page for installation guidelines. If you intend to use a GPU, make sure you install all backend libraries `Theano` needs to run on GPU. Specifically, install `CUDA`, `CuDNN`, `libgpuarray` and `pygpu`.
* [Subword NMT](https://github.com/rsennrich/subword-nmt)
* [fast_align](https://github.com/clab/fast_align) (optional, only for markup handling)
* [MultEval](https://github.com/cidermole/multeval) (optional, only for evaluation)

### Environment variables

`mtrain` uses several environment variables to infer the location of installed tools.

#### Environment variables for Moses backend
The environment variables `MOSES_HOME` and `FASTALIGN_HOME` are used to
locate Moses and fast_align scripts and binaries, respectively. `MOSES_HOME`
should point to the base directory of your Moses installation, containing the
subdirectories `scripts`, `bin`, etc. `FASTALIGN_HOME` is where your compiled
fast_align binaries (`fastalign` and `atools`) are stored.

For evaluation (optional), `mtrain` requires an additional environment variable,
`MULTEVAL_HOME`, a directory containing a MultEval installation and `multeval.sh`, among other things.

To set all of the required variables in your shell session:

```bash
export MOSES_HOME=/path/to/moses
export FASTALIGN_HOME=/path/to/fastalign/bin
export MULTEVAL_HOME=/path/to/multeval
```

#### Environment variables for Nematus backend

_In addition to_ the variables for Moses (see previous section), to train Nematus models you also need: `NEMATUS_HOME` that points to your installation of Nematus, containing the subdirectories `data`, `nematus`, `utils` etc. `SUBWORD_NMT_HOME` should point to your installation of Subword NMT, containing the scripts `learn_bpe.py` and `apply_bpe.py`.

To set all of the required variables in your shell session:

```bash
export MOSES_HOME=/path/to/moses
export FASTALIGN_HOME=/path/to/fastalign/bin
export MULTEVAL_HOME=/path/to/multeval
export NEMATUS_HOME=/path/to/nematus
export SUBWORD_NMT_HOME=/path/to/subword-nmt
```

#### Setting environment variables permanently
If you want the environment variables to be loaded automatically for each of
your shell sessions, simply add the export statements above to your
`~/.bashrc` or `~/.bash_profile`.

#### Environment variables for Python versions
If you experience problems related to Python versions, first check whether you indeed installed both Python 2 and 3. If that does not help, you can also provide to `mtrain` the explicit paths to those Python installations:

```bash
export PYTHON2=/path/to/python2
export PYTHON3=/path/to/python3
```

### Installing the `mtrain` package

For using [pip](https://pypi.python.org/pypi/pip), type (this assumes that `pip` points to your Python 3 installation of pip):

```sh
pip install git+https://github.com/ZurichNLP/mtrain.git
```

or, by cloning the repository:

```sh
git clone https://github.com/ZurichNLP/mtrain.git
cd mtrain
pip install --user .
```

After installation, the main binaries should be available on your system. Just
type `mtrain` or `mtrans` into your console to make sure.

## Basic usage

### Training a Moses model

Engines are trained using the `mtrain` command. Given a parallel corpus located
at `~/my-corpus.en` (source side) and `~/my-corpus.fr` (target side), you can train
a phrase-based Moses English to French engine using the following command:

```sh
mtrain ~/my-corpus en fr
```

This creates the engine in the current working directory. To store it at a
different location, use the `-o` parameter:

```sh
mtrain ~/my-corpus en fr -o ~/my_engine
```

Make sure that the path given to the `-o` parameter is an existing empty
directory on your system.

You may also want to tune your engine using a held-out portion of your parallel
corpus. Using the `-t` (or `--tune`) switch, you can define the number of
segments to be randomly sampled from your parallel corpus and, subsequently,
used to tune the usual model parameters using MERT:

```sh
mtrain ~/my-corpus en fr -o ~/my_engine -t 1000
```

Finally, you may want to train a recaser or truecaser model to steer the
handling of capitalised words in your engine. To do so, just add `-c recasing`
or `-c truecasing` to your command, respectively:

```sh
mtrain ~/my-corpus en fr -o ~/my_engine -t 1000 -c recasing
```

Altogether, this will train a phrase-based (maximum phrase length: 7 tokens) translation
engine with a 5-gram language model (modified Kneser-Ney smoothing) and a
lexicalised (msd-bidirectional-fe) reordering model, as well as a standard Moses
recasing engine. All phrase and reordering tables will be [compressed](http://ufal.mff.cuni.cz/pbml/98/art-junczys-dowmunt.pdf).

### Training a Nematus model

Training with the Nematus backend is similar to the Moses backend. Again, providing a parallel corpus location and source and target language are mandatory for training:

```sh
mtrain ~/my-corpus en fr --backend nematus
```

In order to store the engine in a different location than in the current working directory, use the `-o` parameter:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine
```

Nematus depends on a tuning set, which is used for validation during training.
Therefore, you need to use `-t`/`--tune` to specify the number of segments sampled from your parallel corpus. If the argument for `-t` is not a number, it must be the path to an existing validation set.

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000
```

Specify a casing strategy with the `-c` parameter:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000 -c truecasing
```

Training a Nematus model is best done on GPUs. Use `--device_train` and `--device_validate` to indicate the names of the devices that should be used for training and validation (can be the same as training if the training process does not use all memory on the GPU and devices are not process-exclusive). For CPU, use the name `cpu`. GPU training also benefits from memory preallocation, which you can control with `--preallocate_train` and `--preallocate_validate`.

Here is a full training command:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000 -c truecasing --device_train cuda0 --preallocate_train 0.8 --device_validate cuda1 --preallocate_validate 0.3
```

You may want to perform separately preprocessing data and training, for example when you need to verify the preprocessed data before training. When using `--preprocessing_only` and `--training_only`, you only have to provide the parameters necessary for the respective step.
For example, type:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000 -c truecasing --preprocessing_only
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine --training_only --device_train cuda0 --preallocate_train 0.8 --device_validate cuda1 --preallocate_validate 0.3
```

### Further training options

For advanced options of `mtrain`, type

```sh
mtrain --help
```

### Translation with a trained Moses model

Once training has finished, you can use your engine to translate a sample
segment:

```sh
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr
```

or an entire file:

```sh
mtrans ~/my_engine en fr < my-english-file.txt > french-translation.txt
```

`mtrans` will detect your engine's casing strategy automatically and handle
capitalised words accordingly. If you prefer lowercased output, just add the
`-l` (or `--lowercase`) flag.

### Translation with a trained Nematus model

For using your trained Nematus engine for translating a segment, choose the Nematus backend, device and preallocated memory:

```sh
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr --backend nematus --device_trans cuda0 --preallocate_trans 0.1
```

To translate an entire file, type the command below:

```sh
mtrans ~/my_engine en fr --backend nematus --device_trans cuda0 --preallocate_trans 0.1  < my-english-file.txt > french-translation.txt
```

### Further translation options

For advanced options of `mtrans`, type

```sh
mtrans --help
```

## Advanced usage

### Handling of XML Markup (so far only in Moses backend)

If your training data contains inline markup (for instance, because the training set is extracted from Microsoft Word or XLIFF), then `mtrain` offers several ways of dealing properly with the XML markup. By default, `mtrain` and `mtrans` assume that your data set does not contain any markup and will treat markup as normal tokens.

Handling XML input is controlled by the `--xml_input` option. Here are all possible values for this option, which reflects all the possible strategies for markup handling that are currently implemented:
* `pass-through` ignoring the fact that there is markup (not recommended if your data contains markup),
* `strip` stripping markup before training and translation,
* `strip-reinsert` stripping markup before training and translation. After translation, reinsert markup into the machine-translated segment,
* `mask` in training, replacing markup strings with mask tokens. Before translation, replace markup with mask tokens, "un"-replace mask tokens again in the machine-translated segment.

For more detailed descriptions of those strategies, look [here](http://www.cl.uzh.ch/dam/jcr:e7fb9132-4761-4af4-8f95-7e610a12a705/MA_mathiasmueller_05012017_0008.pdf).

## Troubleshooting

### My Moses model training fails

Make sure to use _absolute_ paths for the `-o` argument (where the trained model will be stored) and the paths to your data sets.
