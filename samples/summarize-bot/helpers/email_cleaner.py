from talon.signature.bruteforce import extract_signature

def clean_email(text):
    cleaned_email, _ = extract_signature(text)
    # cleaned_email = cleaned_email.replace('\n', '').replace('\r', '')
    return cleaned_email