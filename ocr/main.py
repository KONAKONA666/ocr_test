import logging
import os
import click
import pytesseract

from pdf2image import convert_from_path
from pathlib import Path
from logging import getLogger

from .image_processing import *
from .text_processing import *
from .utils import *
from .custom_types import *
from .constants import *

logger = getLogger(__name__)
logging.basicConfig()

PLAIN_PREPROCESS_PIPELINE: List[PipelineFunction] = [
    supersample2x,
    grayscale,
    invert,
    median_blur,
    thresholding,
    erode,
    deskew,
    invert,
]

FORM_PREPROCESS_PIPELINE: List[PipelineFunction] = [
    supersample2x,
    grayscale,
    invert,
    delete_horizontal,
    delete_vertical,
    median_blur,
    thresholding,
    erode,
    deskew,
    invert,
]


def select_pipeline(image: CV2Image) -> List[PipelineFunction]:
    '''
    identifies which pipline should be used for ties image
    based on fraction of horizontal lines, underlines(deleting them significantly improves accuracy).
    :param image:
    :return: List[PipelineFunction]
    '''

    lines = get_fraction_of_lines(image, PLAIN_PREPROCESS_PIPELINE)
    if lines >= 0.01:
        return FORM_PREPROCESS_PIPELINE
    return PLAIN_PREPROCESS_PIPELINE


def preprocess(image: CV2Image) -> Optional[CV2Image]:
    pipeline = select_pipeline(image)
    return apply_pipeline(image, pipeline)


def process(image: CV2Image) -> str:
    ocr_string = pytesseract.image_to_string(image, lang='eng')
    return ocr_string


def postprocess(text: str) -> str:
    clean_text_list = []
    rows = get_rows(text)
    for row in rows:
        tokens = tokenize(row)
        corrected_row = [process_token(token) for token in tokens]
        clean_text_list.append(' '.join(corrected_row))
    return '\n'.join(clean_text_list)


def validate_input_path(ctx, param, value):
    if value.suffix not in SUPPORTED_FORMATS:
        raise click.BadParameter(
            'UNSUPPORTED INPUT FORMAT(supported only .pdf, .png or .jpg)')
    return value


def save(text: str, path: Path) -> None:
    with open(path, 'w') as f:
        f.write(text)


def process_images(files: List[Path]) -> str:
    text = ""
    for path in files:
        logger.info("PROCESSING: {}".format(path))
        image = open_image(path)
        preprocessed_image = preprocess(image)
        raw_text = process(preprocessed_image)
        text += postprocess(raw_text)
    return text


@click.command()
@click.option('-i',
              '--input',
              'input_path',
              required=True,
              callback=validate_input_path,
              type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('-o',
              '--output',
              'output_path',
              required=True,
              type=click.Path(writable=True, path_type=Path))
@click.option('--verbose', is_flag=True, type=bool)
def main(input_path: Path, output_path: Path, verbose: bool) -> None:
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    files = []
    logger.info("STARTING: {}".format(input_path))
    if input_path.suffix == '.pdf':
        pages = convert_from_path(input_path, 500)
        for i, page in enumerate(pages):
            filename = "page-{}.png".format(i)
            page.save(filename, 'PNG')
            files.append(Path(filename))
    else:
        files.append(input_path)
    logger.info("NUM PAGES: {}".format(len(files)))
    text = process_images(files)
    logger.info("TEXT SIZE: {}".format(len(text)))
    save(text, output_path)
    if input_path.suffix == '.pdf':
        for file in files:
            file.unlink()
