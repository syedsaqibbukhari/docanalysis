# docanalysis

Tools for preprocessing scanned images for OCR

[![Build Status](https://travis-ci.org/kba/ocrd_dfkitools.svg?branch=master)](https://travis-ci.org/kba/ocrd_dfkitools)

The sequence of operations is: binarization, deskewing, cropping and dewarping
(or can also be: binarization, dewarping, deskewing, and cropping; depends upon
use-case).

Sample files are available at [OCR-D/assets](https://github.com/OCR-D/ocrd-assets/tree/master/data/dfki-testdata)

## Tools included

### ocrd-anybaseocr-binarize

This function takes a scanned colored /gray scale document image as input and do the black and white binarize image.

Extracted from ocropus-nlbin (from https://github.com/tmbdev/ocropy/).

### ocrd-anybaseocr-deskew

This function takes a document image as input and do the skew correction of that document.

Extracted from ocropus-nlbin (from https://github.com/tmbdev/ocropy/).

### ocrd-anybaseocr-crop

This function takes a document image as input and crops/selects the page
content area only (that's mean remove textual noise as well as any other noise
around page content area)

## Testing

To test the tools, download [OCR-D/assets](https://github.com/OCR-D/assets). In
particular, the code is tested with the
[dfki-testdata](https://github.com/OCR-D/assets/tree/master/data/dfki-testdata)
dataset.

Run `make test` to run all tests.

## License

```
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ```
