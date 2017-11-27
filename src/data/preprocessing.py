from util.tokenize import is_useful, regex_filter
from util import constants

def clean(df):
    df = df[df.apply(lambda x: is_useful(x['text']), axis=1, reduce=True)]
    df['text'] = df['text'].apply(
        lambda t: regex_filter(t, constants.URL_REGEX))  # Remove any unwanted words

    return df