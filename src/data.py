import os
import tarfile
from email import message_from_file
import re
from glob import glob
from pathlib import Path
import pandas as pd
import urllib.request
import urllib.parse
import zipfile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR

def _download_dataset(dest_zip: Path) -> bool:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    try:
        urllib.request.urlretrieve(url, dest_zip)
        return True
    except Exception:
        return False

def _load_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    return df

def _fallback_df() -> pd.DataFrame:
    rows = [
        {"label": "ham", "text": "Hey, are we still meeting later?"},
        {"label": "ham", "text": "Please call me when you arrive."},
        {"label": "spam", "text": "Win a free prize now by texting WIN to 80085"},
        {"label": "spam", "text": "Congratulations, you have won a lottery. Claim now."},
        {"label": "ham", "text": "The report is attached."},
        {"label": "spam", "text": "Free entry in 2 a weekly competition to win cash"},
    ]
    return pd.DataFrame(rows)

class SimpleEmail:
    def __init__(self, subject: str, body: str):
        self.subject = subject
        self.body = body

    @property
    def clean(self):
        sanitizer = '[^A-Za-z]+'
        clean = re.sub(sanitizer, ' ', f'{self.subject or ""} {self.body or ""}')
        clean = clean.lower()
        return re.sub(r'\s+', ' ', clean)

    def __str__(self):
        subject = f'subject: {self.subject}'
        body_first_line = self.body.split('\n')[0]
        body = f'body: {body_first_line}...'
        return f'{subject}\n{body}'

    def __repr__(self):
        return self.__str__()

class EmailIterator:
    def __init__(self, directory: str):
        self._files = glob(f'{directory}/*')
        self._pos = 0

    def __iter__(self):
        self._pos = -1
        return self

    def __next__(self):
        if self._pos < len(self._files) - 1:
            self._pos += 1
            return self.parse_email(self._files[self._pos])
        raise StopIteration()

    @staticmethod
    def parse_email(filename: str) -> SimpleEmail:
        with open(filename,
                  encoding='utf-8',
                  errors='replace') as fp:
            message = message_from_file(fp)

        subject = None
        for item in message.raw_items():
            if item[0] == 'Subject':
                subject = item[1]

        if message.is_multipart():
            body = []
            for b in message.get_payload():
                body.append(str(b))
            body = '\n'.join(body)
        else:
            body = message.get_payload()

        return SimpleEmail(subject, body)

def download_spam_assassin_corpus(dataset_dir: Path):
    base_url = 'https://spamassassin.apache.org'
    corpus_path = 'old/publiccorpus'
    files = {
        '20021010_easy_ham.tar.bz2': 'ham',
        '20021010_hard_ham.tar.bz2': 'ham',
        '20021010_spam.tar.bz2': 'spam',
        '20030228_easy_ham.tar.bz2': 'ham',
        '20030228_easy_ham_2.tar.bz2': 'ham',
        '20030228_hard_ham.tar.bz2': 'ham',
        '20030228_spam.tar.bz2': 'spam',
        '20030228_spam_2.tar.bz2': 'spam',
        '20050311_spam_2.tar.bz2': 'spam'
    }

    downloads_dir = dataset_dir / 'downloads'
    ham_dir = dataset_dir / 'ham'
    spam_dir = dataset_dir / 'spam'

    downloads_dir.mkdir(parents=True, exist_ok=True)
    ham_dir.mkdir(parents=True, exist_ok=True)
    spam_dir.mkdir(parents=True, exist_ok=True)

    for file, spam_or_ham in files.items():
        url = urllib.parse.urljoin(base_url, f'{corpus_path}/{file}')
        tar_filename = downloads_dir / file
        urllib.request.urlretrieve(url, tar_filename)

        with tarfile.open(tar_filename) as tar:
            tar.extractall(path=downloads_dir)
            extracted_root = None
            for tarinfo in tar:
                name = tarinfo.name
                if len(name.split('/')) > 1:
                    directory, filename = name.split('/')
                    if extracted_root is None:
                        extracted_root = directory
                    destination_path = dataset_dir / spam_or_ham / filename
                    if not destination_path.exists():
                        os.rename(downloads_dir / directory / filename, destination_path)
            if extracted_root and (downloads_dir / extracted_root).exists():
                os.rmdir(downloads_dir / extracted_root)

def get_spam_assassin_df() -> pd.DataFrame:
    _ensure_data_dir()
    spam_assassin_data_dir = DATA_DIR / "spam_assassin"
    if not spam_assassin_data_dir.exists() or not (spam_assassin_data_dir / "ham").exists() or not (spam_assassin_data_dir / "spam").exists():
        download_spam_assassin_corpus(spam_assassin_data_dir)

    rows = []
    for email_obj in EmailIterator(str(spam_assassin_data_dir / "ham")):
        rows.append({"label": "ham", "text": email_obj.clean})
    for email_obj in EmailIterator(str(spam_assassin_data_dir / "spam")):
        rows.append({"label": "spam", "text": email_obj.clean})
    
    df = pd.DataFrame(rows)
    return df


def get_sms_spam_df() -> pd.DataFrame:
    _ensure_data_dir()
    zip_path = DATA_DIR / "smsspamcollection.zip"
    if not zip_path.exists():
        ok = _download_dataset(zip_path)
        if not ok:
            return _fallback_df()
    try:
        df = _load_from_zip(zip_path)
    except Exception:
        return _fallback_df()
    return df
