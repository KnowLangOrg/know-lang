from dataclasses import dataclass
from typing import Union, TYPE_CHECKING
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End
from knowlang.utils import create_pydantic_model
from knowlang.agents.unity.nodes.base import (
    UIGenerationState,
    UIGenerationDeps,
    UIGenerationResult,
)


if TYPE_CHECKING:
    from knowlang.agents.unity.nodes.csharp_generator import CSharpGeneratorNode


@dataclass
class USSGeneratorNode(
    BaseNode[UIGenerationState, UIGenerationDeps, UIGenerationResult]
):
    """Node that generates USS (Unity Style Sheets) based on the generated UXML"""

    default_system_prompt = """
You are an expert Unity UI Toolkit developer. Your task is to generate USS (Unity Style Sheets) styling based on the provided UXML markup.

Follow these rules strictly:

1. Generate clean, modern USS styling that follows Unity UI Toolkit best practices
2. Use the class names and IDs from the UXML to create appropriate selectors
3. Include responsive design principles with flexbox layouts
4. Use Unity's color system and design tokens where appropriate
5. Create a cohesive visual design that's modern and user-friendly
6. Include hover states, focus states, and other interactive elements
7. Use appropriate spacing, typography, and visual hierarchy
8. Make sure the styling works well with Unity's UI Toolkit rendering

Common USS properties to use:
- flex-direction, justify-content, align-items for layout
- width, height, min-width, max-width for sizing
- margin, padding for spacing
- background-color, border-color, border-width for visual styling
- font-size, font-weight, color for typography
- border-radius for rounded corners
- box-shadow for depth and elevation
- transition for smooth animations

Example USS structure:
```css
.main-container {
    flex-direction: column;
    padding: 20px;
    background-color: rgb(45, 45, 45);
    min-height: 100%;
}

.title {
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-bottom: 16px;
}

.primary-button {
    background-color: rgb(0, 120, 215);
    color: white;
    padding: 8px 16px;
    border-radius: 4px;
    border-width: 0;
    transition: background-color 0.2s;
}

.primary-button:hover {
    background-color: rgb(0, 100, 180);
}
```

Remember: Generate only the USS content, no explanations or additional text.
"""

    async def run(
        self, ctx: GraphRunContext[UIGenerationState, UIGenerationDeps]
    ) -> Union["CSharpGeneratorNode"]:
        chat_config = ctx.deps.chat_config
        uss_agent = Agent(
            create_pydantic_model(config=chat_config.llm),
            system_prompt=(
                self.default_system_prompt
                if chat_config.llm.system_prompt is None
                else chat_config.llm.system_prompt
            ),
        )

        prompt = f"""
Generate USS styling for the following UXML markup:

{ctx.state.uxml_content}

Requirements:
- Create comprehensive USS styling that matches the UXML structure
- Use the class names and IDs from the UXML
- Create a modern, cohesive visual design
- Include responsive layout with flexbox
- Add hover and focus states for interactive elements
- Use appropriate Unity color schemes and design patterns
- Ensure good visual hierarchy and spacing

Generate only the USS content:
"""

        try:
            result = await uss_agent.run(prompt)
            ctx.state.uss_content = result.output.strip()

            from knowlang.agents.unity.nodes.csharp_generator import CSharpGeneratorNode

            return CSharpGeneratorNode()
        except Exception as e:
            ctx.state.error_message = f"Error generating USS: {e}"
            raise
