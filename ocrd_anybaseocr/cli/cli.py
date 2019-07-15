import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import OcrdAnybaseocrCropper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_deskew import OcrdAnybaseocrDeskewer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_binarize import OcrdAnybaseocrBinarizer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_dewarp import OcrdAnybaseocrDewarper

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_cropping(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrCropper, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_deskew(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDeskewer, *args, **kwargs)    


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_binarize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrBinarizer, *args, **kwargs)        

@click.command()
#@click.option('--pix2pixhd',type=click.Path(), help="Path to pix2pixHD library.",required=True)
@ocrd_cli_options
def ocrd_anybaseocr_dewarp(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDewarper, *args, **kwargs)            