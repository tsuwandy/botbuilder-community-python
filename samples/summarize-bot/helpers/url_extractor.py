from urlextract import URLExtract

def get_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    return urls

def remove_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, ' ')
    text = text.replace('  ', ' ').strip()
    # print(urls)
    return text