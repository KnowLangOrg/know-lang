from dataclasses import dataclass
from typing import Union, TYPE_CHECKING
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext
from knowlang.utils import create_pydantic_model
from knowlang.agents.unity.nodes.base import (
    UIGenerationState,
    UIGenerationDeps,
    UIGenerationResult,
)

if TYPE_CHECKING:
    from knowlang.agents.unity.nodes import UXMLGeneratorNode


@dataclass
class FilenameGeneratorNode(
    BaseNode[UIGenerationState, UIGenerationDeps, UIGenerationResult]
):
    """Node that generates a clean, concise filename based on the UI description"""

    default_system_prompt = """
You are an expert Unity developer and file naming specialist. Your task is to generate a clean, concise filename based on a UI description.

Follow these rules strictly:

1. Generate ONLY a filename - no file extension, no prefix, no additional text
2. Use PascalCase naming convention (e.g., "MainMenu", "InventoryPanel", "SettingsDialog")
3. Make the filename descriptive but concise (2-4 words maximum)
4. Avoid special characters, spaces, or symbols
5. Use meaningful terms that clearly represent the UI's purpose
6. Follow Unity naming conventions for UI elements
7. Consider the UI's primary function and main components
8. Make it easy to understand what the file contains

Examples of good filenames:
- "MainMenu" for a main menu UI
- "InventoryPanel" for an inventory display
- "SettingsDialog" for a settings popup
- "CharacterSheet" for a character information display
- "ShopInterface" for a shopping UI
- "LoginForm" for a login screen
- "GameHUD" for a heads-up display
- "PauseMenu" for a pause menu

Remember: Generate ONLY the filename in PascalCase, nothing else.
"""

    async def run(
        self, ctx: GraphRunContext[UIGenerationState, UIGenerationDeps]
    ) -> Union["UXMLGeneratorNode"]:
        chat_config = ctx.deps.chat_config
        filename_agent = Agent(
            create_pydantic_model(config=chat_config.llm),
            system_prompt=(
                self.default_system_prompt
                if chat_config.llm.system_prompt is None
                else chat_config.llm.system_prompt
            ),
        )

        prompt = f"""
Generate a clean, concise filename for a Unity UI based on this description:

{ctx.state.ui_description}

Requirements:
- Use PascalCase naming convention
- Make it descriptive but concise (2-4 words max)
- No file extension, no prefix, no additional text
- Clearly represent the UI's purpose
- Follow Unity naming conventions

Generate only the filename:
"""

        try:
            result = await filename_agent.run(prompt)
            # Store the generated filename in the state
            ctx.state.filename = result.output.strip()
            
            from knowlang.agents.unity.nodes import UXMLGeneratorNode

            return UXMLGeneratorNode()
        except Exception as e:
            ctx.state.error_message = f"Error generating filename: {e}"
            raise
