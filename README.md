# docanalysis

Tools for preprocessing scanned images for OCR

#Installing

To install anyBaseOCR dependencies system-wide:

    $ sudo pip install -r requirements.txt
    $ sudo python setup.py install

Alternatively, dependencies can be installed into a Virtual Environment:

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    $ python setup.py install

## Tools included

To see how to run binarization, deskew, crop and dewarp method, please follow corresponding below files :


   * [README_binarize.md](./README_binarize.md) instruction for binarization method
   * [README_deskew.md](README_deskew.md) instruction for deskew method
   * [README_cropping.md](README_cropping.md) instruction for cropping method
   * [README_dewarp.md](README_dewarp.md) instruction for dewarp method

## Testing

To test the tools, download [OCR-D/assets](https://github.com/OCR-D/assets). In particular, the code is tested with the
[dfki-testdata](https://github.com/OCR-D/assets/tree/master/data/dfki-testdata) dataset.
