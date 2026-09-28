# pdffigures2.jar: notices and licences

`scripts/fetch_assets.sh` downloads `pdffigures2.jar` into this folder from the
[`assets-v1` release](https://github.com/DavidHir0/kg-extraction/releases/tag/assets-v1).
It is used only by `main.py extract`.

## pdffigures2

Copyright the Allen Institute for Artificial Intelligence.
Licensed under the Apache License, Version 2.0: [`licenses/pdffigures2-Apache-2.0.txt`](licenses/pdffigures2-Apache-2.0.txt).
Source: <https://github.com/allenai/pdffigures2>

The jar was built unmodified from commit
[`3d7ad46`](https://github.com/allenai/pdffigures2/commit/3d7ad46753d4a315cccd1c2bcab398380e88c534)
(version 0.1.0) with `sbt assembly`, because the project publishes no release
jar.

If you use it, please cite:

> Christopher Clark and Santosh Divvala. 2016. PDFFigures 2.0: Mining Figures
> from Research Papers. In *Proceedings of the 16th ACM/IEEE-CS Joint Conference
> on Digital Libraries (JCDL '16)*, 143–152.
> <https://doi.org/10.1145/2910896.2910904>

## Libraries bundled inside the jar

`sbt assembly` packs pdffigures2's dependencies into the jar. Their licences:

| Library | Version | Licence | Licence text |
|---|---|---|---|
| Scala standard library | 2.12.16 | Apache-2.0 | inside the jar: `LICENSE_scala-library`, `NOTICE_scala-library` |
| Apache PDFBox | 2.0.26 | Apache-2.0 | inside the jar: `META-INF/LICENSE_pdfbox-2.0.26`, `META-INF/NOTICE_pdfbox-2.0.26` |
| Apache FontBox | 2.0.26 | Apache-2.0 | inside the jar: `META-INF/LICENSE_fontbox-2.0.26`, `META-INF/NOTICE_fontbox-2.0.26` |
| Apache Commons Logging | 1.2 | Apache-2.0 | inside the jar: `META-INF/LICENSE_commons-logging-1.2.txt`, `META-INF/NOTICE_commons-logging-1.2.txt` |
| spray-json | 1.3.6 | Apache-2.0 | [`licenses/spray-json-Apache-2.0.txt`](licenses/spray-json-Apache-2.0.txt) |
| Typesafe Config | 1.4.2 | Apache-2.0 | [`licenses/typesafe-config-Apache-2.0.txt`](licenses/typesafe-config-Apache-2.0.txt) |
| SLF4J (`slf4j-api`, `jcl-over-slf4j`) | 1.7.36 | MIT (`slf4j-api`), Apache-2.0 (`jcl-over-slf4j`) | [`licenses/slf4j-MIT.txt`](licenses/slf4j-MIT.txt), [`licenses/pdffigures2-Apache-2.0.txt`](licenses/pdffigures2-Apache-2.0.txt) (the Apache-2.0 text) |
| scopt | 4.1.0 | MIT | [`licenses/scopt-MIT.md`](licenses/scopt-MIT.md) |
| Bouncy Castle (`bcprov`, `bcpkix`, `bcmail`, `bcutil`) | 1.71 | Bouncy Castle Licence (MIT) | [`licenses/BouncyCastle.html`](licenses/BouncyCastle.html) |
| Logback (`logback-classic`, `logback-core`) | 1.2.11 | EPL-1.0 or LGPL-2.1, at the licensee's choice | [`licenses/logback.txt`](licenses/logback.txt), [`licenses/EPL-1.0.html`](licenses/EPL-1.0.html) |

Logback is redistributed here under the Eclipse Public License 1.0. Its source
code is available from the Logback project at
<https://github.com/qos-ch/logback/tree/v_1.2.11>.

Each licence text in `licenses/` was taken from the library's own repository at
the bundled version (for `jcl-over-slf4j`, which ships no separate file, the
Apache-2.0 text is reused), and each licence was checked against the library's
Maven POM.
