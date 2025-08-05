from dataclasses import dataclass
from typing import Union, TYPE_CHECKING
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext
from knowlang.utils import create_pydantic_model
from knowlang.agents.unity.nodes.base import UIGenerationState, UIGenerationDeps, UIGenerationResult

if TYPE_CHECKING:
    from knowlang.agents.unity.nodes import USSGeneratorNode


@dataclass
class UXMLGeneratorNode(
    BaseNode[UIGenerationState, UIGenerationDeps, UIGenerationResult]
):
    """Node that generates UXML (Unity UI XML) markup based on the UI description"""

    default_system_prompt = """
You are an expert Unity UI Toolkit developer. Your task is to generate UXML (Unity UI XML) markup based on user descriptions.

Follow these rules strictly:

1. Generate clean, semantic UXML markup that follows Unity UI Toolkit best practices
2. Use appropriate Unity UI elements like:
   - <ui:UXML xmlns:ui="UnityEngine.UIElements">
   - <ui:VisualElement> for containers
   - <ui:Label> for text
   - <ui:Button> for buttons
   - <ui:TextField> for input fields
   - <ui:ScrollView> for scrollable content
   - <ui:ListView> for lists
   - <ui:DropdownField> for dropdowns
   - <ui:Slider> for sliders
   - <ui:Toggle> for checkboxes/toggles
   - <ui:ProgressBar> for progress indicators

3. Always include proper namespaces and class attributes for styling
4. Use meaningful class names that will be referenced in USS
5. Structure the UI logically with proper hierarchy
6. Include appropriate sizing and positioning hints
7. Make the UI responsive and accessible
8. Add proper IDs for elements that will need event binding in C#

Example structure:
```xml
<ui:UXML xmlns:ui="UnityEngine.UIElements">
  <ui:VisualElement name="root" class="main-container">
    <ui:Label text="Title" name="title-label" class="title" />
    <ui:Button text="Click Me" name="action-button" class="primary-button" />
  </ui:VisualElement>
</ui:UXML>
```

Remember: Generate only the UXML content, no explanations or additional text.
"""

    async def run(
        self, ctx: GraphRunContext[UIGenerationState, UIGenerationDeps]
    ) -> Union["USSGeneratorNode"]:
        chat_config = ctx.deps.chat_config
        uxml_agent = Agent(
            create_pydantic_model(config=chat_config.llm),
            system_prompt=(
                self.default_system_prompt
                if chat_config.llm.system_prompt is None
                else chat_config.llm.system_prompt
            ),
        )

        prompt = f"""
Generate UXML markup for the following UI description:

{ctx.state.ui_description}

Requirements:
- Create a complete, valid UXML document
- Use appropriate Unity UI Toolkit elements
- Include meaningful class names for styling
- Add proper IDs for event binding
- Make the UI responsive and accessible

Generate only the UXML content:
"""

        try:
            result = await uxml_agent.run(prompt)
            ctx.state.uxml_content = result.output.strip()

            from knowlang.agents.unity.nodes import USSGeneratorNode

            return USSGeneratorNode()
        except Exception as e:
            ctx.state.error_message = f"Error generating UXML: {e}"
            raise
