import os

import pandas as pd

from pdftabextract.common import read_xml, parse_pages
from pdftabextract.extract import fit_texts_into_grid, datatable_to_dataframe

DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
PICTURE = '.jpg'


def extraxt_table(filepath: str) -> pd.DataFrame:

    table = fit_texts_into_grid(p['texts'], grid)
    return datatable_to_dataframe(table)

