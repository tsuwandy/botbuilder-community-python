
import os

""" Bot Configuration """
class DefaultConfig(object):
    """ Bot Configuration """
    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    GIT_USERNAME = "t-ahmago"
    GIT_PASSWORD = "studioworks_17"