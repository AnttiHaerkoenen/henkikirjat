import os
import argparse

import bs4
import requests
from requests.exceptions import HTTPError, ConnectionError


headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/53.0.2785.143 Safari/537.36',
}
DIGINARC = 'http://digi.narc.fi/digi/'


def download_series_from_na(numbers, folder):
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    print("Downloading...")
    for k, v in numbers.items():
        try:
            url = f"{DIGINARC}fetch_hqjpg.ka?kuid={v}"
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            with open(f'{k}.jpg', 'wb') as fopen:
                for chunk in response.iter_content(100_000):
                    fopen.write(chunk)
            print(f"Page {k} downloaded.")
        except HTTPError:
            print(f"HTTPError occurred at page {k} (kuid: {v})")
        except ConnectionError as ce:
            print(f"Connection error at page {k} (kuid: {v}): {ce}")
            return
    print(f"{len(numbers)} pages downloaded.")


def get_pic_ids(url: str, start_page: int, last_page: int) -> dict:
    """
    Gets kuid values for each archive page based on id.
    :param url: ukid-level page
    :param start_page: first page to get
    :param last_page: inclusive end of page range
    :return: Dict of ints (page: kuid for scanned image files)
    """
    response = requests.get(
        f'{DIGINARC}{url}',
        headers=headers,
        timeout=5,
    )
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, features="html.parser")
    urls = soup.find_all(id='piclink')
    url_ = next(filter(lambda tag: int(tag.string) == start_page, urls))
    url_ = url_.get_attribute_list('href')[0]
    url = f'{DIGINARC}{url_}'
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, features="html.parser")
    options = soup.select('div')[1].select('div')[2].select('form')[0].select('option')
    pages = {
        int(page.string): int(page['value'])
        for page
        in options
        if start_page <= int(page.string) <= last_page
    }
    return pages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to load whole book part from digi.narc.fi")
    parser.add_argument(
        '--url',
        dest='url',
        type=str,
        help=f'Address of containing unit after {DIGINARC}',
    )
    parser.add_argument(
        '--first',
        dest='first',
        type=int,
        help='First page to download. Must be the text in a link.',
    )
    parser.add_argument(
        '--last',
        dest='last',
        type=int,
        help='Last page to download.',
    )
    parser.add_argument(
        '--dir',
        dest='directory',
        type=str,
        help='Where to put downloaded files. Created if doesn\'t exist.',
    )
    args = parser.parse_args()
    pages = get_pic_ids(args.url, args.first, args.last)
    download_series_from_na(pages, args.directory)
