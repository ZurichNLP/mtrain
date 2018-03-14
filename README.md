# mtrain

This Python 3 package provides convenience wrappers to train (`mtrain`) and
translate (`mtrans`) with Moses-based Statistical Machine Translation engines
or Nematus-based Neural Network Translation engines. Given a
parallel corpus of any size, training and translation are as easy as

```sh
mkdir ~/my_engine
mtrain /path/to/my/parallel-corpus en fr --tune 1000 -o ~/my_engine --backend {moses|nematus}
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr --backend {moses|nematus}
```

Installation and further usage instructions are given below. To report a bug or suggest
improvements, please feel free to [open a ticket](https://gitlab.cl.uzh.ch/mt/mtrain/issues).

## Installation

### General requirements
* Python >= 3.5
* [Moses](https://github.com/moses-smt/mosesdecoder) (tested with release 3.0). Make sure to compile with cmph (`./bjam --with-cmph=/path/to/cmph`)

#### Requirements for Moses backend (in addition to general requirements above)
* [fast_align](https://github.com/clab/fast_align)
* [MultEval](https://github.com/cidermole/multeval) (only for evaluation)

#### Requirements for Nematus backend (in addition to general requirements above)
* [Nematus](https://github.com/EdinburghNLP/nematus) including its prerequisites:
  * Python 2.7: For the resulting mixed environment with Python 2.7 (for Nematus) and 3.5 (for Moses), we recommend setting up a virtual environment using [Anaconda3](https://conda.io/docs/user-guide/tasks/manage-python.html)
  * [numpy](https://www.scipy.org/install.html)
  * [Theano](http://deeplearning.net/software/theano) >= 0.7 including dependencies
  * Recommended packages to speed up training: [CUDA](https://developer.nvidia.com/cuda-toolkit-70) >= 7 and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= 4
* [Subword NMT](https://github.com/rsennrich/subword-nmt)

### Environment variables

#### Environment variables for Moses backend
The environment variables `MOSES_HOME` and `FASTALIGN_HOME` are used to
locate Moses and fast_align scripts and binaries, respectively. `MOSES_HOME`
should point to the base directory of your Moses installation, containing the
subdirectories `scripts`, `bin`, etc. `FASTALIGN_HOME` is where your compiled
fast_align binaries (`fastalign` and `atools`) are stored.

For evaluation (optional), `mtrain` requires an additional environment variable,
`MULTEVAL_HOME`, a directory containing a MultEval installation and `multeval.sh`, among other things.

#### Environment variables for Nematus backend
In addition to the environment variable `MOSES_HOME` as described above, `NEMATUS_HOME` and `SUBWORD_NMT_HOME`
are needed to locate its respective scripts and binaries. `NEMATUS_HOME` should point to the Nematus installation,
containing the subdirectories `data`, `nematus`, `utils` etc. `SUBWORD_NMT_HOME` should point to where the 
Subword NMT scripts `learn_bpe.py` and `apply_bpe.py` are stored.

#### Setting environment variables temporarily
To set environment variables for the duration of your shell session,
type (depending on the backend you want to use)

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

#### Special environment variables for Python
Even though we did our best to describe how to set up a mixed environment with Python 2.7 (for Nematus) and 3.5 (for Moses),
this task may be tricky depending on your server environment. To provide some ease, you may use the environment variables
`PYTHON2` and `PYTHON3` if you experience problems. Let the former point to your Python 2 installation and the latter to Python 3.
By doing this, you can ensure that mtrain and mtrans use the respective versions of Python correctly.

```bash
export PYTHON2=/path/to/python2
export PYTHON3=/path/to/python3
```

### Installing the `mtrain` package

For using [pip](https://pypi.python.org/pypi/pip), type

```sh
pip install git+https://gitlab.cl.uzh.ch/laeubli/mtrain.git
```

or clone the repository by typing

```sh
git clone git@gitlab.cl.uzh.ch:mt/mtrain.git
```

After installation, the main binaries should be available on your system. Just
type `mtrain` or `mtrans` into your console to make sure.

IMPORTANT:
You may want to change to the Nematus branch, which contains the latest version of Nematus backend and some bugfixing for Moses backend.
To do so, change to the base directory of the downloaded package and type
```sh
git checkout nematus
```

## Testing of both backends (optional)
In the base directory of the downloaded package, type

```sh
python setup.py test
```

Make sure that the environment variables (see above) are set before running the
tests.

## Basic usage

### Training with Moses backend

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

### Training with Nematus backend

Note: When you try some of the commands below, `mtrain` will insist on you providing more parameters.
This is intentional to introduce you to the Nematus backend step by step.

Training with Nematus backend works quite similar to the Moses backend in regards of the commands you need.
Again, providing a parallel corpus location and source and target language are mandatory for training.

However, for using Nematus you need to specify the respective backend (if not, Moses is used by default):

```sh
mtrain ~/my-corpus en fr --backend nematus
```

In order to store the engine at a different location than in the current working directory, use the `-o` parameter:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine
```

Nematus depends on a tuning set, which is used for repeatedly validating the trained engine.
Therefore, you need to use `-t` and specify the number of segments sampled from your parallel corpus:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000
```

So far, truecasing is the only casing strategy for backend Nematus.
Thus, truecasing is used by default for `-c` and does not have to be specified, but you may if you wish to:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000 -c truecasing
```

Training and the included validation are best performed when using CuDNN to make full use of the GPUs.
You then need to specify which GPU is used for training and validation and how much memory shall be preallocated
for the respective operation. For example, type:

```sh
mtrain ~/my-corpus en fr --backend nematus -o ~/my_engine -t 1000 -c truecasing --device_train cuda0 --preallocate_train 0.8 --device_validate cuda1 --preallocate_validate 0.3
```

You may want to split preprocessing data and training, for example when you need to verify the preprocessed data before training.
When using `--preprocessing_only` and `--training_only`, you only have to provide the parameters necessary for the respective step.
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

### Translation with Moses backend

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

### Translation with Nematus backend

For using your trained Nematus engine for translating a segment, choose the respecive backend, GPU and preallocated memory:

```sh
echo "Consistency is the last refuge of the unimaginative." | mtrans ~/my_engine en fr --backend nematus --device_trans cuda0 --preallocate_trans 0.1
```

To translate an entire file, type the command below.
For translating long texts, we recommend using the parameter `--keep_translation`, which appends a copy of your translation to the file `results.mtrans.txt` in your base directory:

```sh
mtrans ~/my_engine en fr --backend nematus --device_trans cuda0 --preallocate_trans 0.1 --keep_translation < my-english-file.txt
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

## References on Nematus implementation

For implementing backend Nematus, a multitude of repositories, instructions and scripts were used. This list summarises publications and respective instructions and scripts:

Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Edinburgh Neural Machine Translation Systems for WMT 16. In Proceedings of the First Conference on Machine Translation (WMT16). Berlin, Germany. Association for Computational Linguistics.

* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/preprocess.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/train.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/config.py
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/validate.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/postprocess-dev.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/translate.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/sample/postprocess-test.sh
* https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
* https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py

Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondrej Bojar, Alexandra Constantin, and Evan Herbst (2007): Moses: Open Source Toolkit for Statistical Machine Translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics (ACL 2007). Prague, Czech Republic. Association for Computational Linguistics.

* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/train-truecaser.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecase.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/detruecase.perl
* https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl

Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany. Association for Computational Linguistics.

* https://github.com/rsennrich/subword-nmt/blob/master/README.md
* https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py
* https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py

Rico Sennrich, Orhan Firat, Kyunghyun Cho, Alexandra Birch, Barry Haddow, Julian Hitschler, Marcin Junczys-Dowmunt, Samuel Läubli, Antonio Valerio Miceli Barone, Jozef Mokry, and Maria Nadejde (2017): Nematus: a Toolkit for Neural Machine Translation. In Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017). Valencia, Spain, pp. 65-68. European Chapter of the Association for Computational Linguistics.

* https://github.com/EdinburghNLP/nematus/blob/master/README.md
* https://github.com/EdinburghNLP/nematus/blob/master/data/build_dictionary.py
* https://github.com/EdinburghNLP/nematus/blob/master/nematus/nmt.py
* https://github.com/EdinburghNLP/nematus/blob/master/nematus/translate.py

Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller, Dzmitry Bahdanau et al. (2016): Theano: A Python Framework for fast Computation of Mathematical Expressions. CoRR, 1605.02688. DBLP Computer Science Bibliography, University of Trier and Schloss Dagstuhl, Germany.
