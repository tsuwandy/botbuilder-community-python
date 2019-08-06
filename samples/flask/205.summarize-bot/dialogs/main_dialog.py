"""Main dialog. """
from botbuilder.dialogs import (
    ComponentDialog,
    WaterfallDialog,
    WaterfallStepContext,
    DialogTurnResult,
)
from botbuilder.dialogs.prompts import TextPrompt, PromptOptions
from botbuilder.core import MessageFactory
from github_summary_bot import MySummaryBot
from bots import DialogAndWelcomeBot
from .summarize_dialog import SummarizeDialog

class MainDialog(ComponentDialog):
    """Main dialog. """

    def __init__(self, configuration: dict, dialog_id: str = None):
        super(MainDialog, self).__init__(dialog_id or MainDialog.__name__)

        self._configuration = configuration

        self.add_dialog(TextPrompt(TextPrompt.__name__))
        self.add_dialog(SummarizeDialog())
        self.add_dialog(
            WaterfallDialog(
                "WFDialog", [self.intro_step, self.act_step, self.final_step]
            )
        )

        self.initial_dialog_id = "WFDialog"
        self.sum_bot = MySummaryBot()

    async def intro_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Initial prompt."""
        result = self.sum_bot.update_state_reply(step_context.context.activity.text) 
        if (result == ''):
            return await step_context.context.send_activity(DialogAndWelcomeBot.create_welcome_response(step_context.context.activity))

        else:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text(result)
                ),
            )

    async def act_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        # Run the SummarizeDialog, dialog will prompt to find out the remaining details.
        return await step_context.begin_dialog(SummarizeDialog.__name__, self.sum_bot)

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete dialog.
        At this step, display the summary for each comment and summary of all comments
        """
        # If the child dialog ("SummarizeDialog") was cancelled or the user failed
        # to confirm, the Result here will be null.
        if step_context.result is not None:
            result = step_context.result
            await step_context.context.send_activity(MessageFactory.text(result))
        else:
            await step_context.context.send_activity(MessageFactory.text("Thank you."))
        return await step_context.end_dialog()
