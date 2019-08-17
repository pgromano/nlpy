import re
import string


__all__ = [
    "lowercase",
    "redact_phone",
    "redact_money",
    "redact_rates",
    "redact_emails",
    "redact_last_fours",
    "redact_numbers",
    "remove_punctuations",
    "remove_multi_punctuations",
    "remove_multi_tokens",
    "remove_multi_whitespace",
    "pad_punctuations",
]


# Redaction regex patterns
re_redact_phone = re.compile(r"\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}")
re_redact_money = re.compile(r"\$\d+(\.\d{2,})?")
re_redact_rate = re.compile(r"\d*(\.\d+)?\%")
re_redact_email = re.compile(r"[aA-zZ0-9_.+-]+@[aA-zZ0-9-]+\.[aA-zZ0-9-.]+")
re_redact_last_four = re.compile(r"(?<!\d)\*?\d{4}(?!\d)")
re_redact_number = re.compile(r"(?<![-\(\w])\b\d+(?![-\)\w])\b")

# Punctuation regex patterns
re_punct = re.compile(f"([{string.punctuation}])")
re_whitespace = re.compile(f"[{string.whitespace}]")

# Removal regex patterns
re_multi_punct = re.compile(f"([{string.punctuation}])+")
re_multi_token = re.compile(r"\b(\w+)(\s+\1)+\b")


def lowercase(document):
    """ Lowercase text """
    return document.lower()


def redact_phone(document, redact_token="[PHONE]"): 
    """ Redact phone """
    return re_redact_phone.sub(redact_token, document)


def redact_money(document, redact_token="[MONEY]"): 
    """ Redact money """
    return re_redact_money.sub(redact_token, document)


def redact_rates(document, redact_token="[RATE]"): 
    """ Redact rate """
    return re_redact_rate.sub(redact_token, document)


def redact_emails(document, redact_token="[EMAIL]"): 
    """ Redact email """
    return re_redact_email.sub(redact_token, document)


def redact_last_fours(document, redact_token="[CARD]"): 
    """ Redact last four card/id numbers """
    return re_redact_last_four.sub(redact_token, document)


def redact_numbers(document, redact_token="[NUMBER]"): 
    """ Redact number """
    return re_redact_number.sub(redact_token, document)


def remove_punctuations(document):
    """ Remove punctuations """
    return re_punct.sub("", document)


def remove_multi_punctuations(document):
    """ Remove multiple repeat punctuations """
    return re_multi_punct.sub(r"\1", document)


def remove_multi_tokens(document):
    """ Remove multiple repeat tokens """
    return re_multi_token.sub(r"\1", document)


def remove_multi_whitespace(document):
    """ Remove multiple whitespace """
    return re_whitespace.sub(" ", document)

def pad_punctuations(document):
    """ Pad or buffer punctuations """
    return re_punct.sub(r" \1 ", document)