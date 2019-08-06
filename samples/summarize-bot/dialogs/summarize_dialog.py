# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Github.com issues summarization dialog."""

from botbuilder.dialogs import WaterfallDialog, WaterfallStepContext, DialogTurnResult
from botbuilder.dialogs.prompts import ConfirmPrompt, TextPrompt, PromptOptions
from botbuilder.core import MessageFactory
from .cancel_and_help_dialog import CancelAndHelpDialog

class SummarizeDialog(CancelAndHelpDialog):
    """Github.com issues summarization implementation."""

    def __init__(self, dialog_id: str = None):
        super(SummarizeDialog, self).__init__(dialog_id or SummarizeDialog.__name__)

        self.add_dialog(TextPrompt(TextPrompt.__name__))
        # self.add_dialog(ConfirmPrompt(ConfirmPrompt.__name__))
        self.add_dialog(
            WaterfallDialog(
                WaterfallDialog.__name__,
                [
                    self.repo_name_step,
                    self.repo_owner_step,
                    self.summarize_issue_step,
                    self.final_step,
                ],
            )
        )

        self.initial_dialog_id = WaterfallDialog.__name__

    async def repo_name_step(
        self, step_context: WaterfallStepContext
    ) -> DialogTurnResult:
        """Prompt for repo name."""
        sum_bot = step_context.options
        result = sum_bot.update_state_reply(step_context.context.activity.text)
            
        if sum_bot.repo_name is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(sum_bot.questions[1])
                ),
            )  # pylint: disable=line-too-long,bad-continuation
        else:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(result)
                ),
            )  # pylint: disable=line-too-long,bad-continuation

    async def repo_owner_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for repo owner."""
        sum_bot = step_context.options
        result = sum_bot.update_state_reply(step_context.context.activity.text)
        
        # Capture the response to the previous step's prompt
        if sum_bot.owner_name is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(sum_bot.questions[2])
                ),
            )  # pylint: disable=line-too-long,bad-continuation
        else:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(result)
                ),
            )  # pylint: disable=line-too-long,bad-continuation

    async def summarize_issue_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for issue to summarize."""
        sum_bot = step_context.options
        result = sum_bot.update_state_reply(step_context.context.activity.text)
             
        # Capture the results of the previous step
        if sum_bot.issue_header is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(step_context.result)
                ),
            )  # pylint: disable=line-too-long,bad-continuation
        else:
            return await step_context.next(result)

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete the interaction and end the dialog."""
        result = step_context.result
        if result:
            return await step_context.end_dialog(result)
        else:
            return await step_context.end_dialog()