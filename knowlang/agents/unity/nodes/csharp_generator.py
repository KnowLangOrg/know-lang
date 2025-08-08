from dataclasses import dataclass
from typing import Union
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End
from knowlang.utils import create_pydantic_model
from .base import UIGenerationState, UIGenerationDeps, UIGenerationResult


@dataclass
class CSharpGeneratorNode(
    BaseNode[UIGenerationState, UIGenerationDeps, UIGenerationResult]
):
    """Node that generates C# boilerplate code for Unity UI event binding"""

    default_system_prompt = """
You are an expert Unity C# developer. Your task is to generate C# boilerplate code for Unity UI Toolkit event binding based on the provided UXML and USS.

Follow these rules strictly:

1. Generate clean, well-structured C# code that follows Unity best practices
2. Create a class that inherits from VisualElement or MonoBehaviour as appropriate
3. Include proper using statements for Unity UI Toolkit
4. Add field declarations for UI elements that need event binding
5. Implement proper event handlers for interactive elements
6. Use the element names/IDs from the UXML for field names
7. Include comments explaining the purpose of each section
8. Follow C# naming conventions (PascalCase for methods, camelCase for fields)
9. Include proper error handling and null checks
10. Make the code easily extensible for future modifications

Common patterns to include:
- Field declarations for UI elements
- Event handler methods
- Initialize method for setting up event bindings
- Cleanup method for removing event handlers
- Property getters/setters for data binding
- Validation methods where appropriate

Example structure:
```csharp
using UnityEngine;
using UnityEngine.UIElements;

public class GeneratedUI : VisualElement
{
    // UI Element fields
    private Label titleLabel;
    private Button actionButton;
    private TextField inputField;
    
    // Event handlers
    public System.Action<string> OnInputChanged;
    public System.Action OnActionButtonClicked;
    
    public GeneratedUI()
    {
        // Load UXML and USS
        var visualTree = Resources.Load<VisualTreeAsset>("GeneratedUI");
        visualTree.CloneTree(this);
        
        // Get references to UI elements
        titleLabel = this.Q<Label>("title-label");
        actionButton = this.Q<Button>("action-button");
        inputField = this.Q<TextField>("input-field");
        
        // Bind events
        BindEvents();
    }
    
    private void BindEvents()
    {
        if (actionButton != null)
        {
            actionButton.clicked += OnActionButtonClicked;
        }
        
        if (inputField != null)
        {
            inputField.RegisterValueChangedCallback(evt => OnInputChanged?.Invoke(evt.newValue));
        }
    }
    
    public void SetTitle(string title)
    {
        if (titleLabel != null)
        {
            titleLabel.text = title;
        }
    }
    
    public void Cleanup()
    {
        if (actionButton != null)
        {
            actionButton.clicked -= OnActionButtonClicked;
        }
    }
}
```

Remember: Generate only the C# code, no explanations or additional text.
"""

    async def run(
        self, ctx: GraphRunContext[UIGenerationState, UIGenerationDeps]
    ) -> End[UIGenerationResult]:
        chat_config = ctx.deps.chat_config
        csharp_agent = Agent(
            create_pydantic_model(config=chat_config.llm),
            system_prompt=(
                self.default_system_prompt
                if chat_config.llm.system_prompt is None
                else chat_config.llm.system_prompt
            ),
        )

        prompt = f"""
Generate C# boilerplate code for Unity UI Toolkit event binding based on the following UXML and USS:

UXML Content:
{ctx.state.uxml_content}

USS Content:
{ctx.state.uss_content}

UI Description:
{ctx.state.ui_description}

Requirements:
- Create a complete C# class for Unity UI Toolkit
- Include field declarations for all interactive UI elements
- Add proper event handlers and callbacks
- Include initialization and cleanup methods
- Use meaningful class and method names
- Follow Unity and C# best practices
- Make the code easily extensible
- Include proper error handling

Generate only the C# code:
"""

        try:
            result = await csharp_agent.run(prompt)
            ctx.state.csharp_content = result.output.strip()

            return End(
                UIGenerationResult(
                    uxml_content=ctx.state.uxml_content,
                    uss_content=ctx.state.uss_content,
                    csharp_content=ctx.state.csharp_content,
                    ui_description=ctx.state.ui_description,
                )
            )
        except Exception as e:
            ctx.state.error_message = f"Error generating C# code: {e}"
            raise
