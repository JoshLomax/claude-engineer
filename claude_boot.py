"""
Claude Interactive Interface
--------------------------
A command-line tool for interacting with Claude AI model.

Features:
- Code analysis and generation
- Question-answering
- Conversation history tracking
- Token usage monitoring
- Automatic code saving
- Multiple model support

Usage:
    python claude_interface.py [--model MODEL] [--config CONFIG] [--debug]
"""
import os
import sys
import json
import logging
import argparse
import difflib
import time  # Missing import for time.sleep()

from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv 
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

# Set up logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Type aliases
Message = Dict[str, str]
ConversationHistory = List[Message]

# Constants
DEFAULT_CONFIG = {
    "models": [
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-latest"
    ],
    "max_tokens": 4000,
    "temperature": 0.5,
    "log_dir": "logs",
    "code_dir": "generated_code",
    "code_style": {
        "indent": 4,
        "line_length": 88,
        "quote_style": "double",
        "formatting": "black"
    },
    "system_message": """Help the user with their requests. Keep the answers concise and maximise the interpretability of the assistant output."""
}

@dataclass
class AppConfig:
    """Configuration settings for the application."""
    models: List[str]
    max_tokens: int
    temperature: float
    log_dir: str
    code_dir: str
    code_style: Dict[str, Union[int, str]]
    system_message: str

    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Merge default config with loaded config to ensure all fields exist
            merged_config = {**DEFAULT_CONFIG, **config_data}
            cls.validate_config(merged_config)
            return cls(**merged_config)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return cls(**DEFAULT_CONFIG)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {config_path}")
            return cls(**DEFAULT_CONFIG)

    @staticmethod
    def validate_config(config: dict) -> None:
        """Validate configuration fields."""
        required_fields = ['models', 'max_tokens', 'temperature', 'log_dir', 'code_dir', 'code_style']
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
        
class ConsoleUI:
    """Handles all console input/output operations."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.console = Console()
        self.user_color = "green"
        self.assistant_color = "blue"
        self.error_color = "red"
        self.info_color = "yellow"

    def print_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = """
    🤖 Claude Interactive Interface

    Welcome to an intelligent coding and conversation assistant!

    Quick Start Guide:
    • Press Enter three times to finish multi-line input
    • Easily work with .py files or direct code entry
    • Utilize advanced AI model capabilities
    • Explore code analysis, generation, and Q&A

    Tip: Use menu options to navigate different features
    """
        self.console.print(welcome_text, style="bold cyan")

    def print_options(self) -> None:
        table = Table(title="Claude Interactive Interface Options")
        table.add_column("Number", style="cyan")
        table.add_column("Command", style="magenta")
        table.add_column("Description", style="green")
        
        options = [
            ("1", "code", "Submit code for analysis"),
            ("2", "ask", "Ask a question"),
            ("3", "generate", "Generate improved code"),  # Changed from 'modify'
            ("4", "system", "Modify system message"),
            ("5", "model", "Switch AI model"),
            ("6", "record", "Toggle automatic code saving"),
            ("7", "stats", "View token usage statistics"),
            ("8", "quit", "Exit program")
        ]
        
        for number, command, description in options:
            table.add_row(number, command, description)
        
        self.console.print(table)

    def get_input(self, prompt: str, multiline: bool = False) -> str:
        """Get user input with optional multiline support."""
        if not multiline:
            return Prompt.ask(prompt, console=self.console)

        self.console.print(f"\n{prompt}", style=self.user_color)
        self.console.print("(Press Enter three times to finish input)", style=self.info_color)
        lines = []
        empty_lines = 0
        
        while empty_lines < 2:  # Changed from 3 to 2 to match actual behavior
            try:
                line = input()
                if not line:
                    empty_lines += 1
                else:
                    empty_lines = 0
                lines.append(line)
            except EOFError:
                break

        return '\n'.join(lines[:-2])  # Changed from -3 to -2

    def display_response(self, text: str, role: str = "assistant") -> None:
        """Display formatted response with delay between lines."""
        style = self.user_color if role == "user" else self.assistant_color
        for line in text.split('\n'):
            self.console.print(line, style=style)
            time.sleep(0.1)

    def display_error(self, error: str):
        """Display error message."""
        self.console.print(f"Error: {error}", style=self.error_color)

class TokenTracker:
    """Tracks token usage during the session."""
    
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = datetime.now()

    def update(self, input_tokens: int, output_tokens: int):
        """Update token counts."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_stats(self) -> Dict[str, Union[int, str]]:
        """Get current session statistics."""
        duration = datetime.now() - self.start_time
        return {
            "Input tokens": self.input_tokens,
            "Output tokens": self.output_tokens,
            "Total tokens": self.input_tokens + self.output_tokens,
            "Session duration": str(duration).split('.')[0]
        }
    
class ErrorTracker:
    """Tracks and analyzes common code errors."""
    def __init__(self):
        self.error_history = {}
        
    def track_error(self, error_type: str, context: str):
        """Record an error occurrence."""
        if error_type not in self.error_history:
            self.error_history = []
        self.error_history.append({
            'timestamp': datetime.now(),
            'context': context
        })
    
    def generate_report(self) -> Dict:
        """Generate error frequency report."""
        return {
            error_type: len(occurrences)
            for error_type, occurrences in self.error_history.items()
        }
  
class CodeHandler:
    """Handles code-related operations."""
    
    def __init__(self, code_dir: str):
        self.code_dir = Path(code_dir)
        self.code_dir.mkdir(exist_ok=True)

    def save_code(self, code: str, context: str = "") -> Path:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Modified format
            filename = self.code_dir / f"claude_code_{timestamp}.py"

            with open(filename, 'w', encoding='utf-8') as f:
                if context:
                    f.write('\n'.join(f"# {line}" for line in context.split('\n')))
                    f.write('\n\n')
                f.write(code)

            return filename
        except IOError as e:
            raise ValueError(f"Failed to save code: {e}")

    def load_code(self, filename: str) -> str:
        """Load code from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")
        
    def generate_diff(self, original_code: str, fixed_code: str) -> str:
        """Generate a unified diff between original and fixed code."""
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile='original',
            tofile='fixed'
        )
        return ''.join(diff)
    
class TestGenerator:
    """Generates unit tests for Python code."""
    def __init__(self, code_handler: CodeHandler):
        self.code_handler = code_handler
        
    def generate_tests(self, code: str) -> str:
        """Generate unit tests for the given code."""
        test_prompt = (
            "Generate unit tests for this code:\n"
            f"```python\n{code}\n```\n"
            "Include assertions and edge cases."
        )
        # Add Claude API call here to generate tests
        return test_prompt

class BatchCodeHandler:
    """Handles multiple file operations."""

    def __init__(self, code_handler: CodeHandler):
        self.code_handler = code_handler
        
    def process_multiple_files(self, file_paths: List) -> Dict:
        results = {}
        for path in file_paths:
            try:
                code = self.code_handler.load_code(path)
                # Fixed: Store the code with the file path as key
                results = code
            except Exception as e:
                results = f"Error: {str(e)}"
        return results
    
class ClaudeInterface:
    """Main interface for interacting with Claude."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.ui = ConsoleUI(config)
        self.tracker = TokenTracker()
        self.code_handler = CodeHandler(config.code_dir)
        self.client = anthropic.Anthropic()
        self.current_model = config.models[0]
        self.record_mode = False
        self.conversation_history: ConversationHistory = []
        self.batch_handler = BatchCodeHandler(self.code_handler)
        self.test_generator = TestGenerator(self.code_handler)
        self.error_tracker = ErrorTracker()

    def run(self):
        """Main interaction loop."""
        try:
            self.ui.print_welcome()
            while True:
                try:
                    # Only show options if no specific command was provided
                    if len(sys.argv) <= 1:
                        self.ui.print_options()
                    choice = self.ui.get_input("\nWhat would you like to do? ")
                    if choice.lower() in ['quit', '8']:
                        self.show_stats()
                        break
                    self.handle_choice(choice)
                except KeyboardInterrupt:
                    self.ui.display_error("Operation cancelled by user")
                    break
                except Exception as e:
                    self.ui.display_error(str(e))
        finally:
            self.cleanup()

    def handle_choice(self, choice: str):
        """Process user menu choice."""
        choice = choice.lower()
        
        if choice in ['code', '1']:
            self.handle_code_input()
            self.prompt_next_action()
        elif choice in ['ask', '2']:
            self.handle_question_input()
            self.prompt_next_action()
        elif choice in ['generate', '3']:  # Changed from 'modify' to 'generate'
            self.handle_code_generation()
        elif choice in ['system', '4']:
            self.modify_system_message()
        elif choice in ['model', '5']:
            self.switch_model()
        elif choice in ['record', '6']:
            self.toggle_record_mode()
        elif choice in ['stats', '7']:
            self.show_stats()
        elif choice in ['quit', '8']:
            return True
        else:
            self.ui.display_error("Invalid choice")
        return False

    def prompt_next_action(self):
        """Prompt user for next action after a response."""
        self.ui.console.print("\nWhat would you like to do next?", style="yellow")
        self.ui.console.print("1. Ask a follow-up question")
        self.ui.console.print("2. Return to main menu")
        
        choice = self.ui.get_input("Enter your choice (1-2): ")
        
        if choice == "1":
            question = self.ui.get_input("Enter your follow-up question:", multiline=True)
            if question.strip():
                self.send_message(question, is_follow_up=True)
                self.prompt_next_action()  # Recursive call for continuous follow-ups
        elif choice == "2":
            return
        else:
            self.ui.display_error("Invalid choice")
            self.prompt_next_action()

    def handle_code_input(self):
        """Handle code submission."""
        file_input = self.ui.get_input("Enter filename.py or press Enter for direct input: ")

        if file_input.strip():
            try:
                code = self.code_handler.load_code(file_input)
            except ValueError as e:
                self.ui.display_error(str(e))
                code = self.ui.get_input("Enter your code:", multiline=True)
        else:
            code = self.ui.get_input("Enter your code:", multiline=True)

        if not code.strip():
            self.ui.display_error("No code provided")
            return

        message = f"Analyze this Python code:\n```python\n{code}\n```"
        self.send_message(message, hide_input=True)

    def handle_question_input(self):
        """Handle question submission."""
        question = self.ui.get_input("Enter your question:", multiline=True)
        # Check if the question contains a .py file reference
        if ".py" in question:
            try:
                file_path = question.split(".py")[0] + ".py"
                code = self.code_handler.load_code(file_path)
                question = f"{question}\n\nHere's the code from {file_path}:\n```python\n{code}\n```"
            except Exception as e:
                self.ui.display_error(f"Failed to read file: {e}")

        if question.strip():
            # First response to analyze the question
            analysis_message = f"Analyze this question and suggest ways to clarify or expand it: {question}"   
            self.send_message(analysis_message, is_analysis=True)
            
            # Second response to actually answer the question
            self.send_message(question)
        else:
            self.ui.display_error("No question provided")

    def handle_combined_input(self):
        """Handle combined question and code submission."""
        question = self.ui.get_input("Enter your question:", multiline=True)

        # Prompt for file or direct code input
        file_input = self.ui.get_input("Enter filename.py or press Enter for direct code input: ")

        if file_input.strip():
            try:
                code = self.code_handler.load_code(file_input)
            except ValueError as e:
                self.ui.display_error(str(e))
                code = self.ui.get_input("Enter your code:", multiline=True)
        else:
            code = self.ui.get_input("Enter your code:", multiline=True)

        if not (question.strip() and code.strip()):
            self.ui.display_error("Both question and code are required")
            return

        message = f"{question}\n\nHere's the code:\n```python\n{code}\n```"
        self.send_message(message)

    def handle_follow_up(self):
        """Handle follow-up questions."""
        if not self.conversation_history:
            self.ui.display_error("No previous conversation to follow up on")
            return

        question = self.ui.get_input("Enter your follow-up question:", multiline=True)
        if question.strip():
            self.send_message(question, is_follow_up=True)
        else:
            self.ui.display_error("No question provided")

    def send_message(self, message: str, hide_input: bool = False, is_follow_up: bool = False, is_analysis: bool = False):
        """Send message to Claude and handle response.
        
        Args:
            message (str): The message to send
            hide_input (bool): Whether to hide the input in display
            is_follow_up (bool): Whether this is a follow-up question
            is_analysis (bool): Whether this is an analysis request (Now we won't get that pesky error! 🎉)
        """
        messages = [{"role": m["role"], "content": m["content"]} for m in self.conversation_history] if is_follow_up else []
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.messages.create(
                model=self.current_model,
                messages=messages,
                system=self.config.system_message,  # Use the current system message
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            response_text = response.content[0].text

            # Format the response to ensure code blocks are properly displayed
            formatted_response = response_text.replace("```python\n", "```python\n")
            self.ui.display_response(formatted_response)

            # Handle code extraction and saving
            if self.record_mode and "```python" in formatted_response:
                self.extract_and_save_code(formatted_response)

            # Update tracking
            self.tracker.update(
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": formatted_response}
            ])

        except Exception as e:
            self.ui.display_error(f"Failed to get response: {str(e)}")

    def extract_and_save_code(self, response_text: str):
        """Extract and save all code blocks from response."""
        import re
        
        # More robust pattern matching for Python code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response_text, re.DOTALL)
        
        if code_blocks:
            # Combine all code blocks into one file with clear separation
            combined_code = ""
            for i, code_block in enumerate(code_blocks, 1):
                combined_code += f"# Code Block {i}\n"
                combined_code += f"{'#' * 50}\n"
                combined_code += code_block.strip() + "\n\n"
            
            # Save combined code
            filename = self.code_handler.save_code(
                combined_code,
                f"Generated from Claude conversation at {datetime.now()}"
            )
            self.ui.console.print(f"Complete code saved to: {filename}", style="green")

    def modify_system_message(self):
        """Allow user to modify the system message."""
        self.ui.console.print("\nCurrent system message:", style="yellow")
        self.ui.console.print(self.config.system_message)
        
        new_message = self.ui.get_input("\nEnter new system message (or press Enter to cancel):", multiline=True)
        if new_message.strip():
            self.config.system_message = new_message
            self.ui.console.print("System message updated successfully!", style="green")
            # Reset conversation history since context has changed
            self.conversation_history = []
        else:
            self.ui.console.print("System message unchanged", style="yellow")
            
    def switch_model(self):
        """Handle model switching."""
        self.ui.console.print("\nAvailable Models:", style="yellow")
        for i, model in enumerate(self.config.models, 1):
            self.ui.console.print(f"{i}. {model}")

        choice = self.ui.get_input("Select model (number or name): ")
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(self.config.models):
                self.current_model = self.config.models[int(choice) - 1]
            elif choice in self.config.models:
                self.current_model = choice
            else:
                raise ValueError("Invalid model selection")
            
            self.ui.console.print(f"Switched to model: {self.current_model}", style="green")
            self.conversation_history = []  # Reset conversation history
            
        except ValueError as e:
            self.ui.display_error(str(e))

    def toggle_record_mode(self):
        """Toggle automatic code recording."""
        self.record_mode = not self.record_mode
        status = "enabled" if self.record_mode else "disabled"
        self.ui.console.print(f"Record mode {status}", style="green")

    def show_stats(self):
        """Display current session statistics."""
        stats = self.tracker.get_stats()
        self.ui.console.print("\nSession Statistics:", style="yellow")
        for key, value in stats.items():
            self.ui.console.print(f"{key}: {value}")

    def save_session(self):
        """Save session data before exit."""
        if self.conversation_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(self.config.log_dir) / f"session_{timestamp}.json"
            
            data = {
                "timestamp": timestamp,
                "conversation": self.conversation_history,
                "stats": self.tracker.get_stats()
            }
            
            try:
                filename.parent.mkdir(exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                self.ui.console.print(f"\nSession saved to: {filename}", style="green")
            except Exception as e:
                self.ui.display_error(f"Failed to save session: {e}")
                
    @staticmethod
    def validate_config(config: dict) -> None:
        required_fields = ['models', 'max_tokens', 'temperature', 'log_dir', 'code_dir']
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
        
    def cleanup(self):
        """Cleanup resources before exit."""
        try:
            self.save_session()
        except Exception as e:
            self.ui.display_error(f"Cleanup failed: {e}")

    def handle_batch_analysis(self):
        """Handle multiple file analysis."""
        files = self.ui.get_input("Enter Python file paths (comma-separated):").split(',')
        results = self.batch_handler.process_multiple_files()
        for file, result in results.items():
            self.ui.console.print(f"\nAnalyzing {file}:", style="yellow")
            self.handle_code_analysis(result)

    def handle_test_generation(self):
        """Handle unit test generation."""
        code = self.ui.get_input("Enter code or file path:", multiline=True)
        tests = self.test_generator.generate_tests(code)
        self.code_handler.save_code(tests, "generated_tests.py")

    def handle_code_generation(self):
        """Handle code generation and comparison."""
        # Get source code
        file_input = self.ui.get_input("Enter source filename.py or press Enter for direct input: ")
        
        if file_input.strip():
            try:
                original_code = self.code_handler.load_code(file_input)
            except ValueError as e:
                self.ui.display_error(str(e))
                original_code = self.ui.get_input("Enter your original code:", multiline=True)
        else:
            original_code = self.ui.get_input("Enter your original code:", multiline=True)

        # Get requirements for changes
        self.ui.console.print("\nWhat changes would you like to make to the code?", style="yellow")
        self.ui.console.print("Examples:", style="cyan")
        self.ui.console.print("- Add error handling")
        self.ui.console.print("- Optimize performance")
        self.ui.console.print("- Add documentation")
        self.ui.console.print("- Implement new feature")
        
        requirements = self.ui.get_input("\nEnter your requirements:", multiline=True)

        # Generate new code
        message = (
            f"Please generate an improved version of this Python code based on these requirements:\n"
            f"Requirements: {requirements}\n\n"
            f"Original Code:\n```python\n{original_code}\n```\n\n"
            f"Please provide the complete modified code as a single code block."
        )
        
        self.send_message(message, hide_input=True)
        
        # Prompt for output preference
        self.prompt_output_preference(original_code)

    def prompt_output_preference(self, original_code: str):
        """Prompt user for output preference."""
        self.ui.console.print("\nHow would you like to view the changes?", style="bold cyan")
        self.ui.console.print("1. Show only modified sections")
        self.ui.console.print("2. Show complete new code")
        self.ui.console.print("3. Show side-by-side comparison")
        
        choice = self.ui.get_input("Enter your choice (1-3): ")
        
        # Get the last assistant response
        if self.conversation_history:
            new_code = self.extract_code_from_response(self.conversation_history[-1]["content"])
            
            if choice == "1":
                diff = self.code_handler.generate_diff(original_code, new_code)
                self.ui.console.print("\nModified sections:", style="yellow")
                self.ui.console.print(diff)
            elif choice == "2":
                self.ui.console.print("\nComplete new code:", style="yellow")
                self.ui.console.print(new_code)
            elif choice == "3":
                self.show_side_by_side_comparison(original_code, new_code)
            else:
                self.ui.display_error("Invalid choice")

    def extract_code_from_response(self, response: str) -> str:
        """Extract code block from assistant response."""
        import re
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        return code_match.group(1) if code_match else ""

    def show_side_by_side_comparison(self, original_code: str, new_code: str):
        """Display side-by-side comparison of original and new code."""
        from rich.columns import Columns
        
        original_panel = Panel(
            original_code,
            title="Original Code",
            width=60
        )
        new_panel = Panel(
            new_code,
            title="New Code",
            width=60
        )
        
        self.ui.console.print(Columns([original_panel, new_panel]))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Claude Interactive Interface")
    parser.add_argument("--model", help="Default model to use")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main entry point."""
    load_dotenv()
    args = parse_arguments()
    
     # More robust logging configuration
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = AppConfig.from_file(args.config) if args.config else AppConfig(**DEFAULT_CONFIG)
    
    # Override model if specified in arguments
    if args.model:
        config.models.insert(0, args.model)
    
    # Initialize and run interface
    try:
        interface = ClaudeInterface(config)
        interface.run()
    except Exception as e:
        console = Console()
        console.print(f"Fatal error: {str(e)}", style="red")
        if args.debug:
            console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()
