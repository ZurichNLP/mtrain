# mtrain

This python3 package provides convenience wrappers to train (`mtrain`) and
translate with (`mtrans`) Moses-based machine translation engines. Given a
parallel corpus of any size, training and translation are as easy as

```sh
mkdir ~/my_engine
mtrain /path/to/my/parallel-corpus en fr --tune 1000 -o ~/my_engine
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr
```

Installation and usage instructions are given below. To report a bug or suggest
improvements, please feel free to [open a ticket](https://gitlab.cl.uzh.ch/laeubli/mtrain/issues).

## Installation

### Requirements
* Python >= 3.5
* [Moses](https://github.com/moses-smt/mosesdecoder) (tested with release 3.0). Make sure to compile with cmph (`./bjam --with-cmph=/path/to/cmph`)
* [fast_align](https://github.com/clab/fast_align)
* [MultEval](https://github.com/cidermole/multeval) (only for evaluation)

### Environment variables
The environment variables `MOSES_HOME` and `FASTALIGN_HOME` are used to
locate Moses and fast_align scripts and binaries, respectively. `MOSES_HOME`
should point to the base directory of your Moses installation, containing the
subdirectories `scripts`, `bin`, etc. `FASTALIGN_HOME` is where your compiled
fast_align binaries (`fastalign` and `atools`) are stored.

For evaluation (optional), `mtrain` requires an additional environment variable,
`MULTEVAL_HOME`, a directory containing a MultEval installation and `multeval.sh`, among other things.

#### Setting environment variables temporarily
To set `MOSES_HOME` and `FASTALIGN_HOME` for the duration of your shell session,
type

```bash
export MOSES_HOME=/path/to/moses
export FASTALIGN_HOME=/path/to/fastalign/bin
```

And, optionally, also export `MULTEVAL_HOME` by typing

```bash
export MULTEVAL_HOME=/path/to/multeval
```

#### Setting environment variables permanently
If you want the environment variables to be loaded automatically for each of
your shell sessions, simply add the export statements above to your
`~/.bashrc` or `~/.bash_profile`.

### Installation using [pip](https://pypi.python.org/pypi/pip)
```sh
pip install git+https://gitlab.cl.uzh.ch/ZurichNLP/mtrain.git
```
After installation, the main binaries should be available on your system. Just
type `mtrain` or `mtrans` into your console to make sure.

### Testing (optional)
In the base directory of the downloaded package, type
```sh
python setup.py test
```
Make sure that the environment variables (see above) are set before running the
tests.

## Basic usage

### Training

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

For further training options, run

```sh
mtrain --help
```

### Translation

Once training has finished, you can use your engine to translate a sample
segment

```sh
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr
```

or an entire file

```sh
mtrans ~/my_engine en fr < my-english-file.txt > french-translation.txt
```

`mtrans` will detect your engine's casing strategy automatically and handle
capitalised words accordingly. If you prefer lowercased output, just add the
`-l` (or `--lowercase`) flag.

For further translation options, run

```sh
mtrans --help
```

### Handling of XML Markup

If your training data contains inline markup (for instance, because the training set is extracted from Microsoft Word or XLIFF), then `mtrain` offers several ways of dealing properly with the XML markup. By default, `mtrain` and `mtrans` assume that your data set does not contain any markup and will treat markup as normal tokens.

Handling XML input is controlled by the `--xml_input` option. Here are all possible values for this option, which reflects all the possible strategies for markup handling that are currently implemented:
* `pass-through` ignoring the fact that there is markup (not recommended if your data contains markup),
* `strip` stripping markup before training and translation,
* `strip-reinsert` stripping markup before training and translation. After translation, reinsert markup into the machine-translated segment,
* `mask` in training, replacing markup strings with mask tokens. Before translation, replace markup with mask tokens, "un"-replace mask tokens again in the machine-translated segment.

For more detailed descriptions of those strategies, look [here](http://www.cl.uzh.ch/dam/jcr:e7fb9132-4761-4af4-8f95-7e610a12a705/MA_mathiasmueller_05012017_0008.pdf).

## Use cases

Description pending.

In the meantime, you may want to have a look at the advanced options of `mtrain`
and `mtrans` by typing

```sh
mtrain --help
mtrans --help
```
