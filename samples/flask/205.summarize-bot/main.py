# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import http.server
import json
import asyncio
from botbuilder.schema import (Activity, ActivityTypes, Attachment, ChannelAccount)
from botframework.connector import ConnectorClient
from botframework.connector.auth import (MicrosoftAppCredentials,
                                         JwtTokenValidation, SimpleCredentialProvider)
from github_summary_bot import MySummaryBot

APP_ID = ''
APP_PASSWORD = ''
sum_bot = MySummaryBot()

class BotRequestHandler(http.server.BaseHTTPRequestHandler):

    @staticmethod
    def __create_reply_activity(request_activity, text):
        return Activity(
            type=ActivityTypes.message,
            channel_id=request_activity.channel_id,
            conversation=request_activity.conversation,
            recipient=request_activity.from_property,
            from_property=request_activity.recipient,
            text=text,
            service_url=request_activity.service_url)

    def __handle_conversation_update_activity(self, activity):
        self.send_response(202)
        self.end_headers()
        if activity.members_added[0].id != activity.recipient.id:
            credentials = MicrosoftAppCredentials(APP_ID, APP_PASSWORD)
            reply = BotRequestHandler.__create_reply_activity(activity, 'Welcome to Summarize Bot')
            reply.attachments = [self.create_adaptive_card_attachment()]
            connector = ConnectorClient(credentials, base_url=reply.service_url)
            connector.conversations.send_to_conversation(reply.conversation.id, reply)

    def __handle_message_activity(self, activity):
        self.send_response(200)
        self.end_headers()
        credentials = MicrosoftAppCredentials(APP_ID, APP_PASSWORD)
        connector = ConnectorClient(credentials, base_url=activity.service_url)
        #######################
        if sum_bot.state < 0:
            if 'no' in activity.text.lower():
                new_text = 'As you like...., Would you like to start over (yes/no) ?'
                reply = BotRequestHandler.__create_reply_activity(activity, new_text)
            else:
                new_text = sum_bot.update_state_reply(activity.text)
                reply = BotRequestHandler.__create_reply_activity(activity, new_text)
                reply.attachments = [self.create_adaptive_card_attachment()]
        else:
            new_text = sum_bot.update_state_reply(activity.text)
            reply = BotRequestHandler.__create_reply_activity(activity, new_text)

        #######################
        # reply = BotRequestHandler.__create_reply_activity(activity, 'You said: %s' % activity.text)
        connector.conversations.send_to_conversation(reply.conversation.id, reply)

    def __handle_authentication(self, activity):
        credential_provider = SimpleCredentialProvider(APP_ID, APP_PASSWORD)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(JwtTokenValidation.authenticate_request(
                activity, self.headers.get("Authorization"), credential_provider))
            return True
        except Exception as ex:
            self.send_response(401, ex)
            self.end_headers()
            return False
        finally:
            loop.close()

    
    def create_adaptive_card_attachment(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cards/models.json")
        with open(path) as card_file:
            card = json.load(card_file)

        attachment = Attachment(content_type="application/vnd.microsoft.card.adaptive", content=card)
        return attachment

    def __unhandled_activity(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        data = json.loads(str(body, 'utf-8'))
        activity = Activity.deserialize(data)

        if not self.__handle_authentication(activity):
            return

        if activity.type == ActivityTypes.conversation_update.value:
            self.__handle_conversation_update_activity(activity)
        elif activity.type == ActivityTypes.message.value:
            self.__handle_message_activity(activity)
        else:
            self.__unhandled_activity()


try:
    SERVER = http.server.HTTPServer(('localhost', 9000), BotRequestHandler)
    print('Started http server on port 9000, connect to http://localhost:9000/api/messages')
    SERVER.serve_forever()
except KeyboardInterrupt:
    print('^C received, shutting down server')
    SERVER.socket.close()
