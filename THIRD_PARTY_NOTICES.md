# Third-Party Notices

This release contains processed benchmark records derived from lexical-semantic resources. It does not redistribute the original upstream resource packages. The sources used to construct the public benchmark are listed below.

## WordNet 3.0

- Project and license: https://wordnet.princeton.edu/license-and-commercial-use
- Frozen access route: WordNet 3.0 distributed through NLTK 3.9.4

The following notice is reproduced from `wordnet/LICENSE` in the frozen WordNet 3.0 archive:

```text
WordNet Release 3.0

This software and database is being provided to you, the LICENSEE, by
Princeton University under the following license.  By obtaining, using
and/or copying this software and database, you agree that you have
read, understood, and will comply with these terms and conditions.:

Permission to use, copy, modify and distribute this software and
database and its documentation for any purpose and without fee or
royalty is hereby granted, provided that you agree to comply with
the following copyright notice and statements, including the disclaimer,
and that the same appear on ALL copies of the software, database and
documentation, including modifications that you make for internal
use or for distribution.

WordNet 3.0 Copyright 2006 by Princeton University.  All rights reserved.

THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT-
ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE
OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT
INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR
OTHER RIGHTS.

The name of Princeton University or Princeton may not be used in
advertising or publicity pertaining to distribution of the software
and/or database.  Title to copyright in this software, database and
any associated documentation shall at all times remain with
Princeton University and LICENSEE agrees to preserve same.
```

## OpenHowNet

- Source repository: https://github.com/thunlp/OpenHowNet
- Source archive: https://github.com/thunlp/OpenHowNet/archive/refs/heads/master.zip
- Resource archive: https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenHowNet/resources.zip

The source snapshot was obtained from the upstream `master` branch. Its embedded `OpenHowNet/version.py` reports `VERSION = "test"`, which is not treated as a semantic resource version.

OpenHowNet license notice:

```text
MIT License

Copyright (c) 2019 THUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BabelNet-linked information and BabelSememe

- BabelNet project: https://babelnet.org/
- BabelNet Non-Commercial License: https://babelnet.org/full-license
- BabelSememe source repository: https://github.com/thunlp/BabelNet-Sememe-Prediction
- Related paper: Fanchao Qi, Liang Chang, Maosong Sun, Sicong Ouyang, and Zhiyuan Liu. “Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets.” AAAI 2020. https://arxiv.org/abs/1912.01795

The OpenHowNet resource artifact does not encode a verifiable BabelNet version, so this release does not assert one.

The BabelNet-derived portion, and by the approved unified policy the complete benchmark dataset, are distributed under the BabelNet Non-Commercial License described in `DATA_LICENSE.md`.
