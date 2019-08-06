"""Main dialog to welcome users."""
import json
import os.path
from typing import List
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, Attachment, ChannelAccount
from helpers.activity_helper import create_activity_reply
from .dialog_bot import DialogBot

class DialogAndWelcomeBot(DialogBot):
    """Main dialog to welcome users implementation."""

    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            # Greet anyone that was not the target (recipient) of this message.
            # To learn more about Adaptive Cards, see https://aka.ms/msbot-adaptivecards
            # for more details.
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(self.create_welcome_response(turn_context.activity))

    @staticmethod
    def create_welcome_response(activity: Activity):
        """Create welcome response with card."""
        welcome_card = DialogAndWelcomeBot.create_adaptive_card_attachment()
        return DialogAndWelcomeBot.create_response(activity, welcome_card, 'Welcome to Summarize Bot')
        
    @staticmethod
    def create_response(activity: Activity, attachment: Attachment, text: str):
        """Create an attachment message response."""
        response = create_activity_reply(activity, text)
        response.attachments = [attachment]
        return response

    @staticmethod
    def create_adaptive_card_attachment():
        """Create an adaptive card."""
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..\cards\models.json")
        with open(path) as card_file:
            card = json.load(card_file)
        
        return Attachment(
            content_type="application/vnd.microsoft.card.adaptive", content=card)
